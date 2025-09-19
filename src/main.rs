use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{unbounded, Receiver, Sender, TryRecvError};
use dashmap::{DashMap, DashSet};
use eframe::egui;
use egui::{Color32, RichText, Vec2};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use rand::Rng;
use rayon::prelude::*;
use serde::Serialize;

// u64 キー専用のノーハッシュ（高速化）
use nohash_hasher::BuildNoHashHasher;
type U64Map<V> = std::collections::HashMap<u64, V, BuildNoHashHasher<u64>>;
type U64Set = std::collections::HashSet<u64, BuildNoHashHasher<u64>>;
type DU64Map<V> = DashMap<u64, V, BuildNoHashHasher<u64>>;
type DU64Set = DashSet<u64, BuildNoHashHasher<u64>>;

// ====== 盤面定数 ======
const W: usize = 6;
const H: usize = 14;
const MASK14: u16 = (1u16 << H) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    Blank,     // '.'
    Any,       // 'N' (空白 or 色)
    Any4,      // 'X' (色のみ)
    Abs(u8),   // 0..12 = 'A'..'M'
    Fixed(u8), // 0..=3 = '0'..'3' (RGBY固定)
}

#[inline(always)]
fn apply_clear_no_fall(pre: &[[u16; W]; 4], clear: &[u16; W]) -> [[u16; W]; 4] {
    let mut out = [[0u16; W]; 4];
    let mut keep = [0u16; W];
    for (x, k) in keep.iter_mut().enumerate() {
        *k = (!clear[x]) & MASK14;
    }
    for (out_col, pre_col) in out.iter_mut().zip(pre.iter()) {
        for (&k, (cell, &p)) in keep.iter().zip(out_col.iter_mut().zip(pre_col.iter())) {
            *cell = p & k;
        }
    }
    out
}

fn draw_pair_preview(ui: &mut egui::Ui, pair: (u8, u8)) {
    let sz = Vec2::new(18.0, 18.0);
    ui.vertical(|ui| {
        // 表示規約: デフォルトは軸ぷよの上に子ぷよ（axis below, child above 表示）
        let (txt1, fill1, stroke1) = cell_style(Cell::Fixed(pair.1)); // child (上)
        let (txt0, fill0, stroke0) = cell_style(Cell::Fixed(pair.0)); // axis (下)
        let top = egui::Button::new(RichText::new(txt1).size(11.0))
            .min_size(sz)
            .fill(fill1)
            .stroke(stroke1);
        let bot = egui::Button::new(RichText::new(txt0).size(11.0))
            .min_size(sz)
            .fill(fill0)
            .stroke(stroke0);
        let _ = ui.add(top);
        let _ = ui.add(bot);
    });
}

// ---- 計測（DFS 深さ×フェーズ） -------------------------

#[derive(Default, Clone, Copy)]
struct DfsDepthTimes {
    gen_candidates: Duration,
    assign_cols: Duration,
    upper_bound: Duration,

    // 葉専用
    leaf_fall_pre: Duration,
    leaf_hash: Duration,
    leaf_memo_get: Duration,          // 今回の最適化後はほぼ0のまま
    leaf_memo_miss_compute: Duration, // 到達判定（reaches_t...）
    out_serialize: Duration,
}
#[derive(Default, Clone, Copy)]
struct DfsDepthCounts {
    nodes: u64,
    cand_generated: u64,
    pruned_upper: u64,
    leaves: u64,
    // 葉早期リターン（落下や到達判定より前）
    leaf_pre_tshort: u64,        // 4T 未満で不可能
    leaf_pre_e1_impossible: u64, // E1 不可能（4連結なし）
    memo_lhit: u64,              // 以降は基本0（残しつつ非使用）
    memo_ghit: u64,
    memo_miss: u64,
}
#[derive(Default, Clone)]
struct ProfileTotals {
    dfs_times: [DfsDepthTimes; W + 1],
    dfs_counts: [DfsDepthCounts; W + 1],
    io_write_total: Duration,
}
#[derive(Default, Clone)]
struct TimeDelta {
    dfs_times: [DfsDepthTimes; W + 1],
    dfs_counts: [DfsDepthCounts; W + 1],
    io_write_total: Duration,
}

impl ProfileTotals {
    fn add_delta(&mut self, d: &TimeDelta) {
        for i in 0..=W {
            let a = &mut self.dfs_times[i];
            let b = &d.dfs_times[i];
            a.gen_candidates += b.gen_candidates;
            a.assign_cols += b.assign_cols;
            a.upper_bound += b.upper_bound;
            a.leaf_fall_pre += b.leaf_fall_pre;
            a.leaf_hash += b.leaf_hash;
            a.leaf_memo_get += b.leaf_memo_get;
            a.leaf_memo_miss_compute += b.leaf_memo_miss_compute;
            a.out_serialize += b.out_serialize;

            let ac = &mut self.dfs_counts[i];
            let bc = &d.dfs_counts[i];
            ac.nodes += bc.nodes;
            ac.cand_generated += bc.cand_generated;
            ac.pruned_upper += bc.pruned_upper;
            ac.leaves += bc.leaves;
            ac.leaf_pre_tshort += bc.leaf_pre_tshort;
            ac.leaf_pre_e1_impossible += bc.leaf_pre_e1_impossible;
            ac.memo_lhit += bc.memo_lhit;
            ac.memo_ghit += bc.memo_ghit;
            ac.memo_miss += bc.memo_miss;
        }
        self.io_write_total += d.io_write_total;
    }
}

fn time_delta_has_any(d: &TimeDelta) -> bool {
    if d.io_write_total != Duration::ZERO {
        return true;
    }
    for i in 0..=W {
        let t = d.dfs_times[i];
        if t.gen_candidates != Duration::ZERO
            || t.assign_cols != Duration::ZERO
            || t.upper_bound != Duration::ZERO
            || t.leaf_fall_pre != Duration::ZERO
            || t.leaf_hash != Duration::ZERO
            || t.leaf_memo_get != Duration::ZERO
            || t.leaf_memo_miss_compute != Duration::ZERO
            || t.out_serialize != Duration::ZERO
        {
            return true;
        }
        let c = d.dfs_counts[i];
        if c.nodes != 0
            || c.cand_generated != 0
            || c.pruned_upper != 0
            || c.leaves != 0
            || c.leaf_pre_tshort != 0
            || c.leaf_pre_e1_impossible != 0
            || c.memo_lhit != 0
            || c.memo_ghit != 0
            || c.memo_miss != 0
        {
            return true;
        }
    }
    false
}

// 計測マクロ：enabled 時のみ計測
macro_rules! prof {
    ($enabled:expr, $slot:expr, $e:expr) => {{
        if $enabled {
            let __t0 = Instant::now();
            let __r = $e;
            $slot += __t0.elapsed();
            __r
        } else {
            $e
        }
    }};
}

// ---- ここまで計測 ---------------------------------------

#[derive(Serialize)]
#[allow(dead_code)]
struct OutputLine {
    key: String,
    hash: u32,
    chains: u32,
    pre_chain_board: Vec<String>,
    example_mapping: HashMap<char, u8>,
    mirror: bool,
}

// ====== GUI アプリ状態 ======
struct App {
    board: Vec<Cell>, // y*W+x, y=0が最下段
    threshold: u32,
    lru_k: u32,
    out_path: Option<std::path::PathBuf>,
    out_name: String,

    // 早期終了（進捗停滞）
    stop_progress_plateau: f32, // 0..=1

    // 切替: 4個消しモード（5個以上で消去が起きた瞬間に除外）
    exact_four_only: bool,

    // 計測 ON/OFF
    profile_enabled: bool,

    // 実行状態
    running: bool,
    abort_flag: Arc<AtomicBool>,
    rx: Option<Receiver<Message>>,
    stats: Stats,
    preview: Option<[[u16; W]; 4]>,
    log_lines: Vec<String>,

    // 画面モード
    mode: Mode,
    // 連鎖生成モードの状態
    cp: ChainPlay,
}

#[derive(Default, Clone)]
struct Stats {
    searching: bool,
    unique: u64,
    output: u64,
    nodes: u64,
    pruned: u64,
    memo_hit_local: u64,
    memo_hit_global: u64,
    memo_miss: u64,
    total: BigUint,
    done: BigUint,
    rate: f64,
    memo_len: usize,
    lru_limit: usize,

    // 計測結果合計（UI は終了/停止時に表示）
    profile: ProfileTotals,
}

#[derive(Clone, Copy, Default)]
struct StatDelta {
    nodes: u64,
    leaves: u64,
    outputs: u64,
    pruned: u64,
    lhit: u64,
    ghit: u64,
    mmiss: u64,
}

enum Message {
    Log(String),
    Preview([[u16; W]; 4]),
    Progress(Stats),
    Finished(Stats),
    Error(String),
    // 追加：時間の増分メッセージ
    TimeDelta(TimeDelta),
}

impl Default for App {
    fn default() -> Self {
        let mut board = vec![Cell::Blank; W * H];
        // デフォルトは「下3段をN」
        for y in 0..3 {
            for x in 0..W {
                board[y * W + x] = Cell::Any;
            }
        }
        Self {
            board,
            threshold: 7,
            lru_k: 300,
            out_path: None,
            out_name: "results.jsonl".to_string(),
            stop_progress_plateau: 0.0, // 無効（0.10などにすると有効）
            exact_four_only: false,
            profile_enabled: false,
            running: false,
            abort_flag: Arc::new(AtomicBool::new(false)),
            rx: None,
            stats: Stats::default(),
            preview: None,
            log_lines: vec!["待機中".into()],
            mode: Mode::BruteForce,
            cp: ChainPlay::default(),
        }
    }
}

// 画面モード
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    BruteForce,
    ChainPlay,
}

// 連鎖生成モード：保存状態
#[derive(Clone, Copy)]
struct SavedState {
    cols: [[u16; W]; 4],
    pair_index: usize,
}

// アニメーション段階
#[derive(Clone, Copy, PartialEq, Eq)]
enum AnimPhase {
    AfterErase,
    AfterFall,
}

#[derive(Clone, Copy)]
struct AnimState {
    phase: AnimPhase,
    since: Instant,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Orient {
    Up,
    Right,
    Down,
    Left,
}

#[derive(Clone, Copy)]
struct CpMove {
    x: usize,
    orient: Orient,
}

#[derive(Clone, Copy, Debug, Default)]
struct CpSearchStatus {
    depth_limit: usize,
    depth: usize,
    branch_index: usize,
    branch_count: usize,
}

enum CpSearchMessage {
    DepthStart { depth_limit: usize },
    Progress(CpSearchStatus),
    Found { depth: usize, mv: CpMove },
    Failed,
    Cancelled,
}

// 連鎖生成モードのワーク
struct ChainPlay {
    cols: [[u16; W]; 4],
    pair_seq: Vec<(u8, u8)>, // 軸, 子（0..=3）
    pair_index: usize,
    target_chain: u32,
    undo_stack: Vec<SavedState>,
    anim: Option<AnimState>,
    lock: bool, // 連鎖アニメ中ロック
    // アニメ表示用：消去直後の盤面と、落下後の次盤面
    erased_cols: Option<[[u16; W]; 4]>,
    next_cols: Option<[[u16; W]; 4]>,
    search_rx: Option<Receiver<CpSearchMessage>>,
    search_thread: Option<thread::JoinHandle<()>>,
    search_cancel: Option<Arc<AtomicBool>>,
    search_status: Option<CpSearchStatus>,
    search_depth_limit: Option<usize>,
    pending_target: Option<u32>,
}

impl Default for ChainPlay {
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        let mut seq = Vec::with_capacity(128);
        for _ in 0..128 {
            seq.push((rng.gen_range(0..4), rng.gen_range(0..4)));
        }
        let cols = [[0u16; W]; 4];
        let s0 = SavedState {
            cols,
            pair_index: 0,
        };
        Self {
            cols,
            pair_seq: seq,
            pair_index: 0,
            target_chain: 3,
            undo_stack: vec![s0],
            anim: None,
            lock: false,
            erased_cols: None,
            next_cols: None,
            search_rx: None,
            search_thread: None,
            search_cancel: None,
            search_status: None,
            search_depth_limit: None,
            pending_target: None,
        }
    }
}

#[inline(always)]
fn orient_symbol(orient: Orient) -> &'static str {
    match orient {
        Orient::Up => "↑",
        Orient::Right => "→",
        Orient::Down => "↓",
        Orient::Left => "←",
    }
}

#[inline(always)]
fn col_height_from_cols(cols: &[[u16; W]; 4], x: usize) -> usize {
    debug_assert!(x < W);
    let occ = (cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) & MASK14;
    occ.count_ones() as usize
}

#[inline(always)]
fn calc_col_heights(cols: &[[u16; W]; 4]) -> [usize; W] {
    let mut heights = [0usize; W];
    for x in 0..W {
        heights[x] = col_height_from_cols(cols, x);
    }
    heights
}

#[inline(always)]
fn count_valid_moves(heights: &[usize; W]) -> usize {
    let mut total = 0usize;
    for x in 0..W {
        let h = heights[x];
        if h >= H {
            continue;
        }
        if h + 1 < H {
            total += 2; // Up + Down
        }
        if x + 1 < W && heights[x + 1] < H {
            total += 1; // Right
        }
        if x >= 1 && heights[x - 1] < H {
            total += 1; // Left
        }
    }
    total
}

#[inline(always)]
fn set_cell_in_cols(cols: &mut [[u16; W]; 4], x: usize, y: usize, color: u8) {
    debug_assert!(x < W && y < H);
    let bit = 1u16 << y;
    let idx = (color as usize).min(3);
    cols[idx][x] |= bit;
}

fn simulate_move_on_cols(
    cols: &[[u16; W]; 4],
    heights: &[usize; W],
    pair: (u8, u8),
    mv: CpMove,
) -> Option<(u32, [[u16; W]; 4])> {
    let mut next = *cols;
    match mv.orient {
        Orient::Up => {
            let h = heights[mv.x];
            if h + 1 >= H {
                return None;
            }
            set_cell_in_cols(&mut next, mv.x, h, pair.0);
            set_cell_in_cols(&mut next, mv.x, h + 1, pair.1);
        }
        Orient::Down => {
            let h = heights[mv.x];
            if h + 1 >= H {
                return None;
            }
            set_cell_in_cols(&mut next, mv.x, h, pair.1);
            set_cell_in_cols(&mut next, mv.x, h + 1, pair.0);
        }
        Orient::Right => {
            if mv.x + 1 >= W {
                return None;
            }
            let h0 = heights[mv.x];
            let h1 = heights[mv.x + 1];
            if h0 >= H || h1 >= H {
                return None;
            }
            set_cell_in_cols(&mut next, mv.x, h0, pair.0);
            set_cell_in_cols(&mut next, mv.x + 1, h1, pair.1);
        }
        Orient::Left => {
            if mv.x == 0 {
                return None;
            }
            let h0 = heights[mv.x];
            let h1 = heights[mv.x - 1];
            if h0 >= H || h1 >= H {
                return None;
            }
            set_cell_in_cols(&mut next, mv.x, h0, pair.0);
            set_cell_in_cols(&mut next, mv.x - 1, h1, pair.1);
        }
    }

    let mut chain = 0u32;
    loop {
        let clear = compute_erase_mask_cols(&next);
        let has_clear = (0..W).any(|x| clear[x] != 0);
        if !has_clear {
            break;
        }
        chain += 1;
        next = apply_given_clear_and_fall(&next, &clear);
    }

    Some((chain, next))
}

fn cp_run_target_search(
    cols: [[u16; W]; 4],
    pair_seq: Vec<(u8, u8)>,
    start_index: usize,
    target: u32,
    tx: Sender<CpSearchMessage>,
    cancel: Arc<AtomicBool>,
) {
    if pair_seq.is_empty() {
        let _ = tx.send(CpSearchMessage::Failed);
        return;
    }
    let seq_len = pair_seq.len();
    for depth_limit in 1..=seq_len {
        if cancel.load(Ordering::Relaxed) {
            let _ = tx.send(CpSearchMessage::Cancelled);
            return;
        }
        if tx
            .send(CpSearchMessage::DepthStart { depth_limit })
            .is_err()
        {
            return;
        }
        if let Some((depth, mv)) = cp_search_depth_limit(
            cols,
            &pair_seq,
            start_index,
            depth_limit,
            target,
            &tx,
            &cancel,
        ) {
            let _ = tx.send(CpSearchMessage::Found { depth, mv });
            return;
        }
    }
    if cancel.load(Ordering::Relaxed) {
        let _ = tx.send(CpSearchMessage::Cancelled);
    } else {
        let _ = tx.send(CpSearchMessage::Failed);
    }
}

fn cp_search_depth_limit(
    cols: [[u16; W]; 4],
    pair_seq: &[(u8, u8)],
    start_index: usize,
    depth_limit: usize,
    target: u32,
    tx: &Sender<CpSearchMessage>,
    cancel: &AtomicBool,
) -> Option<(usize, CpMove)> {
    if depth_limit == 0 || pair_seq.is_empty() {
        return None;
    }
    let seq_len = pair_seq.len();
    let pair = pair_seq[start_index];
    let heights = calc_col_heights(&cols);
    let total = count_valid_moves(&heights);
    if total == 0 {
        let status = CpSearchStatus {
            depth_limit,
            depth: 1,
            branch_index: 0,
            branch_count: 0,
        };
        let _ = tx.send(CpSearchMessage::Progress(status));
        return None;
    }
    let mut branch_index = 0usize;
    let try_move = |mv: CpMove, branch_index: usize| -> Option<(usize, CpMove)> {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        let status = CpSearchStatus {
            depth_limit,
            depth: 1,
            branch_index,
            branch_count: total,
        };
        let _ = tx.send(CpSearchMessage::Progress(status));
        if let Some((chain, next_cols)) = simulate_move_on_cols(&cols, &heights, pair, mv) {
            if chain >= target {
                return Some((1, mv));
            }
            if depth_limit > 1 {
                let next_index = (start_index + 1) % seq_len;
                if let Some(rest) = cp_search_recursive(
                    next_cols,
                    pair_seq,
                    next_index,
                    depth_limit - 1,
                    target,
                    tx,
                    cancel,
                    depth_limit,
                    2,
                ) {
                    return Some((1 + rest, mv));
                }
            }
        }
        None
    };
    for x in 0..W {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        let h = heights[x];
        if h >= H {
            continue;
        }
        if h + 1 < H {
            branch_index += 1;
            if let Some(found) = try_move(
                CpMove {
                    x,
                    orient: Orient::Up,
                },
                branch_index,
            ) {
                return Some(found);
            }
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            branch_index += 1;
            if let Some(found) = try_move(
                CpMove {
                    x,
                    orient: Orient::Down,
                },
                branch_index,
            ) {
                return Some(found);
            }
        }
        if x + 1 < W && heights[x + 1] < H {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            branch_index += 1;
            if let Some(found) = try_move(
                CpMove {
                    x,
                    orient: Orient::Right,
                },
                branch_index,
            ) {
                return Some(found);
            }
        }
        if x >= 1 && heights[x - 1] < H {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            branch_index += 1;
            if let Some(found) = try_move(
                CpMove {
                    x,
                    orient: Orient::Left,
                },
                branch_index,
            ) {
                return Some(found);
            }
        }
    }
    None
}

fn cp_search_recursive(
    cols: [[u16; W]; 4],
    pair_seq: &[(u8, u8)],
    pair_index: usize,
    depth_remaining: usize,
    target: u32,
    tx: &Sender<CpSearchMessage>,
    cancel: &AtomicBool,
    depth_limit_total: usize,
    current_depth: usize,
) -> Option<usize> {
    if depth_remaining == 0 || pair_seq.is_empty() {
        return None;
    }
    let seq_len = pair_seq.len();
    let pair = pair_seq[pair_index];
    let heights = calc_col_heights(&cols);
    let total = count_valid_moves(&heights);
    if total == 0 {
        let status = CpSearchStatus {
            depth_limit: depth_limit_total,
            depth: current_depth,
            branch_index: 0,
            branch_count: 0,
        };
        let _ = tx.send(CpSearchMessage::Progress(status));
        return None;
    }
    let mut branch_index = 0usize;
    let try_move = |mv: CpMove, branch_index: usize| -> Option<usize> {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        let status = CpSearchStatus {
            depth_limit: depth_limit_total,
            depth: current_depth,
            branch_index,
            branch_count: total,
        };
        let _ = tx.send(CpSearchMessage::Progress(status));
        if let Some((chain, next_cols)) = simulate_move_on_cols(&cols, &heights, pair, mv) {
            if chain >= target {
                return Some(1);
            }
            if depth_remaining > 1 {
                let next_index = (pair_index + 1) % seq_len;
                if let Some(rest) = cp_search_recursive(
                    next_cols,
                    pair_seq,
                    next_index,
                    depth_remaining - 1,
                    target,
                    tx,
                    cancel,
                    depth_limit_total,
                    current_depth + 1,
                ) {
                    return Some(1 + rest);
                }
            }
        }
        None
    };
    for x in 0..W {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        let h = heights[x];
        if h >= H {
            continue;
        }
        if h + 1 < H {
            branch_index += 1;
            if let Some(found) = try_move(
                CpMove {
                    x,
                    orient: Orient::Up,
                },
                branch_index,
            ) {
                return Some(found);
            }
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            branch_index += 1;
            if let Some(found) = try_move(
                CpMove {
                    x,
                    orient: Orient::Down,
                },
                branch_index,
            ) {
                return Some(found);
            }
        }
        if x + 1 < W && heights[x + 1] < H {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            branch_index += 1;
            if let Some(found) = try_move(
                CpMove {
                    x,
                    orient: Orient::Right,
                },
                branch_index,
            ) {
                return Some(found);
            }
        }
        if x >= 1 && heights[x - 1] < H {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            branch_index += 1;
            if let Some(found) = try_move(
                CpMove {
                    x,
                    orient: Orient::Left,
                },
                branch_index,
            ) {
                return Some(found);
            }
        }
    }
    None
}

fn install_japanese_fonts(ctx: &egui::Context) {
    use egui::{FontData, FontDefinitions, FontFamily};

    let mut fonts = FontDefinitions::default();

    // Windows フォント候補（存在したものを最初に採用）
    let windir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".to_string());
    let fontdir = std::path::Path::new(&windir).join("Fonts");
    let candidates = [
        "meiryo.ttc",   // Meiryo
        "meiryob.ttc",  // Meiryo UI（環境による）
        "YuGothR.ttc",  // 游ゴシック（Regular）
        "YuGothM.ttc",  // 游ゴシック（Medium）
        "YuGothB.ttc",  // 游ゴシック（Bold）
        "YuGothUI.ttc", // 游ゴシック UI
        "YuGothU.ttc",  // 旧表記の可能性
        "msgothic.ttc", // MS ゴシック（最終手段）
        "msmincho.ttc", // MS 明朝（最終手段）
    ];

    let mut loaded = false;
    for name in candidates.iter() {
        let path = fontdir.join(name);
        if let Ok(bytes) = std::fs::read(&path) {
            let key = format!("jp-{}", name.to_lowercase());
            fonts
                .font_data
                .insert(key.clone(), FontData::from_owned(bytes));
            fonts
                .families
                .get_mut(&FontFamily::Proportional)
                .unwrap()
                .insert(0, key.clone());
            fonts
                .families
                .get_mut(&FontFamily::Monospace)
                .unwrap()
                .insert(0, key.clone());
            loaded = true;
            break;
        }
    }

    if loaded {
        ctx.set_fonts(fonts);
    } else {
        eprintln!(
            "日本語フォントを見つけられませんでした。C:\\Windows\\Fonts を確認してください。"
        );
    }
}

// ====== eframe エントリ ======
fn main() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(egui::vec2(980.0, 760.0)),
        ..Default::default()
    };

    eframe::run_native(
        "ぷよぷよ 連鎖形 総当たり（6×14）— Rust GUI",
        options,
        Box::new(|cc| {
            install_japanese_fonts(&cc.egui_ctx);
            Box::new(App::default())
        }),
    )
    .map_err(|e| anyhow!("GUI起動に失敗: {e}"))
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 連鎖アニメーション進行
        if self.mode == Mode::ChainPlay {
            self.cp_step_animation();
        }
        self.cp_poll_search();
        // 受信メッセージ処理（安全：take→処理→必要なら戻す）
        if let Some(rx) = self.rx.take() {
            let mut keep_rx = true;

            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Message::Log(s) => self.push_log(s),
                    Message::Preview(p) => self.preview = Some(p),
                    Message::Progress(mut st) => {
                        let prof = self.stats.profile.clone();
                        st.profile = prof;
                        self.stats = st;
                    }
                    Message::Finished(mut st) => {
                        let prof = self.stats.profile.clone();
                        st.profile = prof;
                        self.stats = st;
                        self.running = false;
                        self.abort_flag.store(false, Ordering::Relaxed);
                        self.push_log("完了".into());
                        keep_rx = false;
                    }
                    Message::Error(e) => {
                        self.running = false;
                        self.abort_flag.store(false, Ordering::Relaxed);
                        self.push_log(format!("エラー: {e}"));
                        keep_rx = false;
                    }
                    Message::TimeDelta(td) => {
                        self.stats.profile.add_delta(&td);
                    }
                }
            }

            if keep_rx {
                self.rx = Some(rx);
            }
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.heading("ぷよぷよ 連鎖形 総当たり（6×14）— Rust GUI（列ストリーミング＋LRU形キャッシュ＋並列化＋計測＋追撃最適化）");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.mode, Mode::BruteForce, "総当たり");
                ui.selectable_value(&mut self.mode, Mode::ChainPlay, "連鎖生成");
            });
        });

        // 左ペイン全体をひとつの ScrollArea でまとめる
        egui::SidePanel::left("left").min_width(420.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.spacing_mut().item_spacing = Vec2::new(8.0, 8.0);
                if self.mode == Mode::BruteForce {
                    // ── 入力と操作（総当たり） ─────────────────────────────
                    ui.group(|ui| {
                        ui.label("入力と操作");
                        ui.label("左クリック: A→B→…→M / 中クリック: N→X→N / 右クリック: ・（空白） / Shift+左: RGBY");

                        ui.horizontal_wrapped(|ui| {
                            ui.add(egui::DragValue::new(&mut self.threshold).clamp_range(1..=19).speed(0.1));
                            ui.label("連鎖閾値");
                            ui.add_space(8.0);
                            ui.add(egui::DragValue::new(&mut self.lru_k).clamp_range(10..=1000).speed(1.0));
                            ui.label("形キャッシュ上限(千)");
                        });

                        // ★ 進捗停滞 早期終了
                        ui.horizontal_wrapped(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.stop_progress_plateau)
                                    .clamp_range(0.0..=1.0)
                                    .speed(0.01),
                            );
                            ui.label("早期終了: 進捗停滞比 (0=無効, 例 0.10)");
                        });

                        ui.horizontal_wrapped(|ui| {
                            ui.checkbox(&mut self.exact_four_only, "4個消しモード（5個以上で消えたら除外）");
                        });

                        ui.horizontal_wrapped(|ui| {
                            ui.checkbox(&mut self.profile_enabled, "計測を有効化（軽量）");
                        });

                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(!self.running, egui::Button::new("Run"))
                                .clicked()
                            {
                                self.start_run();
                            }
                            if ui
                                .add_enabled(self.running, egui::Button::new("Stop"))
                                .clicked()
                            {
                                self.abort_flag.store(true, Ordering::Relaxed);
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label("出力ファイル:");
                            ui.text_edit_singleline(&mut self.out_name);
                            if ui.button("Browse…").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .set_title("保存先の選択")
                                    .set_file_name(&self.out_name)
                                    .save_file()
                                {
                                    self.out_path = Some(path.clone());
                                    self.out_name = path
                                        .file_name()
                                        .unwrap_or_default()
                                        .to_string_lossy()
                                        .into();
                                }
                            }
                        });
                    });
                } else {
                    // ── 連鎖生成モードの操作 ─────────────────────────────
                    ui.group(|ui| {
                        ui.label("連鎖生成 — 操作");
                        let cur = self.cp_current_pair();
                        let nxt = self.cp_next_pair();
                        let dnx = self.cp_dnext_pair();

                        ui.horizontal(|ui| {
                            ui.label("現在手:");
                            draw_pair_preview(ui, cur);
                            ui.add_space(12.0);
                            ui.label("Next:");
                            draw_pair_preview(ui, nxt);
                            ui.add_space(12.0);
                            ui.label("Next2:");
                            draw_pair_preview(ui, dnx);
                        });

                        ui.add_space(6.0);
                        let searching = self.cp_search_in_progress();
                        let can_ops = !self.cp.lock && self.cp.anim.is_none() && !searching;
                        ui.horizontal(|ui| {
                            ui.label("目標連鎖数");
                            ui.add_enabled(
                                !searching,
                                egui::DragValue::new(&mut self.cp.target_chain)
                                    .clamp_range(1u32..=19u32)
                                    .speed(1.0),
                            );
                            if ui
                                .add_enabled(can_ops, egui::Button::new("目標配置"))
                                .clicked()
                            {
                                self.cp_place_target();
                            }
                        });
                        ui.horizontal(|ui| {
                            if ui.add_enabled(can_ops, egui::Button::new("ランダム配置")).clicked() {
                                self.cp_place_random();
                            }
                            if ui.add_enabled(can_ops && self.cp.undo_stack.len() > 1, egui::Button::new("戻る")).clicked() {
                                self.cp_undo();
                            }
                            if ui.add_enabled(can_ops && self.cp.undo_stack.len() > 1, egui::Button::new("初手に戻る")).clicked() {
                                self.cp_reset_to_initial();
                            }
                        });
                        if searching {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                if let Some(status) = self.cp.search_status {
                                    if status.branch_count > 0 {
                                        ui.label(format!(
                                            "探索中: 深さ制限{} / {}手目 ({} / {} 通り)",
                                            status.depth_limit,
                                            status.depth,
                                            status.branch_index,
                                            status.branch_count
                                        ));
                                    } else {
                                        ui.label(format!(
                                            "探索中: 深さ制限{} / {}手目 (有効手なし)",
                                            status.depth_limit,
                                            status.depth
                                        ));
                                    }
                                } else if let Some(depth) = self.cp.search_depth_limit {
                                    ui.label(format!("探索中: 深さ制限{} を走査中…", depth));
                                } else if let Some(target) = self.cp.pending_target {
                                    ui.label(format!("目標{}連鎖を探索中…", target));
                                } else {
                                    ui.label("探索中…");
                                }
                            });
                        } else {
                            ui.label(if self.cp.lock { "連鎖中…（操作ロック）" } else { "待機中" });
                        }
                    });
                }

                ui.separator();

                // ── 処理時間（累積） ────────────────────────────────────────
                if self.mode == Mode::BruteForce && !self.running && self.profile_enabled && has_profile_any(&self.stats.profile) {
                    ui.group(|ui| {
                        ui.label("処理時間（累積）");
                        show_profile_table(ui, &self.stats.profile);
                    });
                    ui.separator();
                }

                // ── プレビュー ─────────────────────────────────────────────
                if self.mode == Mode::BruteForce {
                    ui.label("プレビュー（E1直前の落下後盤面）");
                    ui.add_space(4.0);
                    if let Some(cols) = &self.preview {
                        draw_preview(ui, cols);
                    } else {
                        ui.label(
                            egui::RichText::new("（実行中に更新表示）")
                                .italics()
                                .color(Color32::GRAY),
                        );
                    }
                }

                ui.separator();

                if self.mode == Mode::BruteForce {
                    // ── ログ ──────────────────────────────────────────────
                    ui.label("ログ");
                    for line in &self.log_lines {
                        ui.monospace(line);
                    }
                }

                ui.separator();

                if self.mode == Mode::BruteForce {
                    // ── 実行・進捗 ────────────────────────────────────────
                    ui.group(|ui| {
                        ui.label("実行・進捗");
                        let pct = {
                            let total = self.stats.total.to_f64().unwrap_or(0.0);
                            let done = self.stats.done.to_f64().unwrap_or(0.0);
                            if total > 0.0 { (done / total * 100.0).clamp(0.0, 100.0) } else { 0.0 }
                        };
                        ui.label(format!("進捗: {:.1}%", pct));
                        ui.add(egui::ProgressBar::new((pct / 100.0) as f32).show_percentage());

                        ui.add_space(4.0);
                        ui.monospace(format!(
                            "探索中: {} / 代表集合: {} / 出力件数: {} / 展開節点: {} / 枝刈り: {} / 総組合せ(厳密): {} / 完了: {} / 速度: {:.1} nodes/s / メモ: L-hit={} G-hit={} Miss={} / 形キャッシュ: 上限={} 実={}",
                            if self.stats.searching { "YES" } else { "NO" },
                            self.stats.unique,
                            self.stats.output,
                            self.stats.nodes,
                            self.stats.pruned,
                            &self.stats.total,
                            &self.stats.done,
                            self.stats.rate,
                            self.stats.memo_hit_local,
                            self.stats.memo_hit_global,
                            self.stats.memo_miss,
                            self.stats.lru_limit,
                            self.stats.memo_len
                        ));
                    });
                }
            });
        });

        // 盤面側もスクロール可能に
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if self.mode == Mode::BruteForce {
                        ui.label("盤面（左: A→B→…→M / 中: N↔X / 右: ・ / Shift+左: RGBY）");
                        ui.add_space(6.0);

                        let cell_size = Vec2::new(28.0, 28.0);
                        let gap = 2.0;

                        let shift_pressed = ui.input(|i| i.modifiers.shift);
                        for y in (0..H).rev() {
                            ui.horizontal(|ui| {
                                for x in 0..W {
                                    let i = y * W + x;
                                    let (text, fill, stroke) = cell_style(self.board[i]);
                                    let btn = egui::Button::new(RichText::new(text).size(12.0))
                                        .min_size(cell_size)
                                        .fill(fill)
                                        .stroke(stroke);
                                    let resp = ui.add(btn);
                                    if resp.clicked_by(egui::PointerButton::Primary) {
                                        if shift_pressed {
                                            self.board[i] = cycle_fixed(self.board[i]);
                                        } else {
                                            self.board[i] = cycle_abs(self.board[i]);
                                        }
                                    }
                                    if resp.clicked_by(egui::PointerButton::Middle) {
                                        self.board[i] = cycle_any(self.board[i]);
                                    }
                                    if resp.clicked_by(egui::PointerButton::Secondary) {
                                        self.board[i] = Cell::Blank;
                                    }
                                    ui.add_space(gap);
                                }
                            });
                            ui.add_space(gap);
                        }
                    } else {
                        ui.label("連鎖生成 — 盤面");
                        ui.add_space(6.0);
                        draw_preview(ui, &self.cp.cols);
                    }
                });
        });

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

// 連鎖生成モード：実装（App メソッド）
impl App {
    // ===== 連鎖生成モード：ユーティリティ =====
    fn cp_current_pair(&self) -> (u8, u8) {
        let idx = self.cp.pair_index % self.cp.pair_seq.len().max(1);
        self.cp.pair_seq[idx]
    }
    fn cp_next_pair(&self) -> (u8, u8) {
        let idx = (self.cp.pair_index + 1) % self.cp.pair_seq.len().max(1);
        self.cp.pair_seq[idx]
    }
    fn cp_dnext_pair(&self) -> (u8, u8) {
        let idx = (self.cp.pair_index + 2) % self.cp.pair_seq.len().max(1);
        self.cp.pair_seq[idx]
    }

    fn cp_search_in_progress(&self) -> bool {
        self.cp.search_thread.is_some()
    }

    fn cp_undo(&mut self) {
        if self.cp_search_in_progress() {
            return;
        }
        if self.cp.undo_stack.len() > 1 && !self.cp.lock {
            self.cp.anim = None;
            self.cp.erased_cols = None;
            self.cp.next_cols = None;
            if let Some(prev) = self.cp.undo_stack.pop() {
                let _ = prev; // pop current snapshot
            }
            if let Some(last) = self.cp.undo_stack.last().copied() {
                self.cp.cols = last.cols;
                self.cp.pair_index = last.pair_index;
            }
        }
    }

    fn cp_reset_to_initial(&mut self) {
        if self.cp_search_in_progress() {
            self.cp_cancel_search();
        }
        if self.cp.lock {
            return;
        }
        self.cp.anim = None;
        self.cp.erased_cols = None;
        self.cp.next_cols = None;
        if let Some(first) = self.cp.undo_stack.first().copied() {
            self.cp.cols = first.cols;
            self.cp.pair_index = first.pair_index;
            // 初期スナップショットだけ残す
            self.cp.undo_stack.clear();
            self.cp.undo_stack.push(first);
        }
        self.cp.lock = false;
    }

    fn cp_cancel_search(&mut self) {
        if let Some(cancel) = &self.cp.search_cancel {
            cancel.store(true, Ordering::Relaxed);
        }
    }

    fn cp_finish_search_thread(&mut self) {
        if let Some(handle) = self.cp.search_thread.take() {
            let _ = handle.join();
        }
        self.cp.search_cancel = None;
    }

    fn cp_handle_search_found(&mut self, depth: usize, mv: CpMove) {
        // 事前スナップショット
        self.cp.undo_stack.push(SavedState {
            cols: self.cp.cols,
            pair_index: self.cp.pair_index,
        });

        let pair = self.cp_current_pair();
        self.cp_place_with(mv.x, mv.orient, pair);
        self.cp_check_and_start_chain();

        let target = self
            .cp
            .pending_target
            .take()
            .unwrap_or_else(|| self.cp.target_chain.max(1));
        self.push_log(format!(
            "目標{}連鎖: 深さ{}で列{}{}に配置",
            target,
            depth,
            mv.x + 1,
            orient_symbol(mv.orient)
        ));
    }

    fn cp_handle_search_failure(&mut self) {
        let target = self
            .cp
            .pending_target
            .take()
            .unwrap_or_else(|| self.cp.target_chain.max(1));
        self.push_log(format!(
            "目標{}連鎖に到達する配置が見つかりませんでした",
            target
        ));
    }

    fn cp_handle_search_cancel(&mut self) {
        if let Some(target) = self.cp.pending_target.take() {
            self.push_log(format!("目標{}連鎖の探索を中断しました", target));
        } else {
            self.push_log("目標連鎖の探索を中断しました".into());
        }
    }

    fn cp_poll_search(&mut self) {
        if let Some(rx) = self.cp.search_rx.take() {
            let mut keep_rx = true;
            let mut finalize = false;
            let mut receiver = Some(rx);
            loop {
                let Some(current) = receiver.as_ref() else {
                    break;
                };
                match current.try_recv() {
                    Ok(CpSearchMessage::DepthStart { depth_limit }) => {
                        self.cp.search_depth_limit = Some(depth_limit);
                        self.cp.search_status = None;
                    }
                    Ok(CpSearchMessage::Progress(status)) => {
                        self.cp.search_depth_limit = Some(status.depth_limit);
                        self.cp.search_status = Some(status);
                    }
                    Ok(CpSearchMessage::Found { depth, mv }) => {
                        self.cp.search_status = None;
                        self.cp.search_depth_limit = None;
                        self.cp_handle_search_found(depth, mv);
                        receiver = None;
                        keep_rx = false;
                        finalize = true;
                        break;
                    }
                    Ok(CpSearchMessage::Failed) => {
                        self.cp.search_status = None;
                        self.cp.search_depth_limit = None;
                        self.cp_handle_search_failure();
                        receiver = None;
                        keep_rx = false;
                        finalize = true;
                        break;
                    }
                    Ok(CpSearchMessage::Cancelled) => {
                        self.cp.search_status = None;
                        self.cp.search_depth_limit = None;
                        self.cp_handle_search_cancel();
                        receiver = None;
                        keep_rx = false;
                        finalize = true;
                        break;
                    }
                    Err(TryRecvError::Empty) => {
                        break;
                    }
                    Err(TryRecvError::Disconnected) => {
                        self.cp.search_status = None;
                        self.cp.search_depth_limit = None;
                        self.cp.pending_target = None;
                        self.push_log("目標連鎖探索が異常終了しました".into());
                        receiver = None;
                        keep_rx = false;
                        finalize = true;
                        break;
                    }
                }
            }

            if keep_rx {
                if let Some(rx) = receiver {
                    self.cp.search_rx = Some(rx);
                }
            }

            if finalize {
                self.cp_finish_search_thread();
            }
        }
    }

    fn cp_place_target(&mut self) {
        if self.cp.lock || self.cp.anim.is_some() || self.cp_search_in_progress() {
            return;
        }
        if self.cp.pair_seq.is_empty() {
            self.push_log("手順が設定されていません".into());
            return;
        }

        let target = self.cp.target_chain.max(1);
        let len = self.cp.pair_seq.len();
        let start_idx = self.cp.pair_index % len;
        let cols = self.cp.cols;
        let seq = self.cp.pair_seq.clone();
        let (tx, rx) = unbounded();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_thread = Arc::clone(&cancel);
        let handle = thread::spawn(move || {
            cp_run_target_search(cols, seq, start_idx, target, tx, cancel_thread);
        });

        self.cp.search_rx = Some(rx);
        self.cp.search_thread = Some(handle);
        self.cp.search_cancel = Some(cancel);
        self.cp.search_status = None;
        self.cp.search_depth_limit = None;
        self.cp.pending_target = Some(target);

        self.push_log(format!("目標{}連鎖の探索を開始", target));
    }

    fn cp_place_random(&mut self) {
        if self.cp.lock || self.cp.anim.is_some() || self.cp_search_in_progress() {
            return;
        }
        // 事前スナップショット
        self.cp.undo_stack.push(SavedState {
            cols: self.cp.cols,
            pair_index: self.cp.pair_index,
        });

        let pair = self.cp_current_pair();
        let mut rng = rand::thread_rng();

        // 有効手集合を列挙
        let mut moves: Vec<(usize, Orient)> = Vec::new();
        for x in 0..W {
            // 垂直（Up/Down）: 同一列に2個置けるか
            let h = self.cp_col_height(x);
            if h + 1 < H {
                moves.push((x, Orient::Up));
                moves.push((x, Orient::Down));
            }
            // 右
            if x + 1 < W {
                let h0 = self.cp_col_height(x);
                let h1 = self.cp_col_height(x + 1);
                if h0 < H && h1 < H {
                    moves.push((x, Orient::Right));
                }
            }
            // 左
            if x >= 1 {
                let h0 = self.cp_col_height(x);
                let h1 = self.cp_col_height(x - 1);
                if h0 < H && h1 < H {
                    moves.push((x, Orient::Left));
                }
            }
        }
        if moves.is_empty() {
            // 置けない（詰み）: スナップショットは維持するがインデックスは進めない
            self.push_log("置ける場所がありません".into());
            let _ = self.cp.undo_stack.pop();
            return;
        }
        let (x, orient) = moves[rng.gen_range(0..moves.len())];

        self.cp_place_with(x, orient, pair);
        // 連鎖開始チェック
        self.cp_check_and_start_chain();
    }

    fn cp_place_with(&mut self, x: usize, orient: Orient, pair: (u8, u8)) {
        // 盤面に2つの色を追加（重力後の最下段に直接置く）
        match orient {
            Orient::Up => {
                let h = self.cp_col_height(x);
                self.cp_set_cell(x, h, pair.0);
                self.cp_set_cell(x, h + 1, pair.1);
            }
            Orient::Down => {
                let h = self.cp_col_height(x);
                self.cp_set_cell(x, h, pair.1);
                self.cp_set_cell(x, h + 1, pair.0);
            }
            Orient::Right => {
                let h0 = self.cp_col_height(x);
                let h1 = self.cp_col_height(x + 1);
                self.cp_set_cell(x, h0, pair.0);
                self.cp_set_cell(x + 1, h1, pair.1);
            }
            Orient::Left => {
                let h0 = self.cp_col_height(x);
                let h1 = self.cp_col_height(x - 1);
                self.cp_set_cell(x, h0, pair.0);
                self.cp_set_cell(x - 1, h1, pair.1);
            }
        }
        // 次の手へ
        if !self.cp.pair_seq.is_empty() {
            self.cp.pair_index = (self.cp.pair_index + 1) % self.cp.pair_seq.len();
        }
    }

    fn cp_col_height(&self, x: usize) -> usize {
        col_height_from_cols(&self.cp.cols, x)
    }

    fn cp_set_cell(&mut self, x: usize, y: usize, color: u8) {
        if x >= W || y >= H {
            return;
        }
        set_cell_in_cols(&mut self.cp.cols, x, y, color);
    }

    fn cp_check_and_start_chain(&mut self) {
        // 4連結抽出
        let clear = compute_erase_mask_cols(&self.cp.cols);
        let any = (0..W).any(|x| clear[x] != 0);
        if !any {
            // 連鎖なし
            return;
        }
        // 連鎖開始：操作ロック
        self.cp.lock = true;
        // 消去直後表示と次盤面作成
        let erased = apply_clear_no_fall(&self.cp.cols, &clear);
        let next = apply_given_clear_and_fall(&self.cp.cols, &clear);
        self.cp.erased_cols = Some(erased);
        self.cp.next_cols = Some(next);
        self.cp.cols = erased; // 消えた状態を表示
        self.cp.anim = Some(AnimState {
            phase: AnimPhase::AfterErase,
            since: Instant::now(),
        });
    }

    fn cp_step_animation(&mut self) {
        let Some(anim) = self.cp.anim else {
            return;
        };
        let elapsed = anim.since.elapsed();
        if elapsed < Duration::from_millis(500) {
            return;
        }
        match anim.phase {
            AnimPhase::AfterErase => {
                if let Some(next) = self.cp.next_cols.take() {
                    self.cp.cols = next;
                }
                self.cp.anim = Some(AnimState {
                    phase: AnimPhase::AfterFall,
                    since: Instant::now(),
                });
                // 次の消去準備は AfterFall 経由
            }
            AnimPhase::AfterFall => {
                // 次の連鎖チェック
                let clear = compute_erase_mask_cols(&self.cp.cols);
                let any = (0..W).any(|x| clear[x] != 0);
                if !any {
                    // 完了
                    self.cp.anim = None;
                    self.cp.erased_cols = None;
                    self.cp.next_cols = None;
                    self.cp.lock = false;
                } else {
                    let erased = apply_clear_no_fall(&self.cp.cols, &clear);
                    let next = apply_given_clear_and_fall(&self.cp.cols, &clear);
                    self.cp.cols = erased;
                    self.cp.erased_cols = Some(erased);
                    self.cp.next_cols = Some(next);
                    self.cp.anim = Some(AnimState {
                        phase: AnimPhase::AfterErase,
                        since: Instant::now(),
                    });
                }
            }
        }
    }
}

// ===== App ユーティリティ =====
impl App {
    fn push_log(&mut self, s: String) {
        self.log_lines.push(s);
        if self.log_lines.len() > 500 {
            let cut = self.log_lines.len() - 500;
            self.log_lines.drain(0..cut);
        }
    }

    fn start_run(&mut self) {
        // 準備
        let threshold = self.threshold.clamp(1, 19);
        let lru_limit = (self.lru_k.clamp(10, 1000) as usize) * 1000;
        let outfile = if let Some(p) = &self.out_path {
            p.clone()
        } else {
            std::path::PathBuf::from(&self.out_name)
        };
        // 盤面を文字配列へ
        let board_chars: Vec<char> = self.board.iter().map(|c| c.label_char()).collect();

        let (tx, rx) = unbounded::<Message>();
        self.rx = Some(rx);
        self.running = true;
        self.preview = None;
        self.log_lines.clear();
        self.stats.profile = ProfileTotals::default();

        let abort = self.abort_flag.clone();
        abort.store(false, Ordering::Relaxed);

        // 停滞比
        let stop_progress_plateau = self.stop_progress_plateau.clamp(0.0, 1.0);
        let exact_four_only = self.exact_four_only;
        let profile_enabled = self.profile_enabled;

        self.push_log(format!(
            "出力: JSONL / 形キャッシュ上限 ≈ {} 形 / 保存先: {} / 進捗停滞比={:.2} / 4個消しモード={} / 計測={}",
            lru_limit,
            outfile.display(),
            stop_progress_plateau,
            if exact_four_only { "ON" } else { "OFF" },
            if profile_enabled { "ON" } else { "OFF" },
        ));

        thread::spawn(move || {
            if let Err(e) = run_search(
                board_chars,
                threshold,
                lru_limit,
                outfile,
                tx.clone(),
                abort,
                stop_progress_plateau,
                exact_four_only,
                profile_enabled,
            ) {
                let _ = tx.send(Message::Error(format!("{e:?}")));
            }
        });
    }
}

fn has_profile_any(p: &ProfileTotals) -> bool {
    if p.io_write_total != Duration::ZERO {
        return true;
    }
    for i in 0..=W {
        let t = p.dfs_times[i];
        if t.gen_candidates != Duration::ZERO
            || t.assign_cols != Duration::ZERO
            || t.upper_bound != Duration::ZERO
            || t.leaf_fall_pre != Duration::ZERO
            || t.leaf_hash != Duration::ZERO
            || t.leaf_memo_get != Duration::ZERO
            || t.leaf_memo_miss_compute != Duration::ZERO
            || t.out_serialize != Duration::ZERO
        {
            return true;
        }
        let c = p.dfs_counts[i];
        if c.nodes != 0
            || c.cand_generated != 0
            || c.pruned_upper != 0
            || c.leaves != 0
            || c.leaf_pre_tshort != 0
            || c.leaf_pre_e1_impossible != 0
            || c.memo_lhit != 0
            || c.memo_ghit != 0
            || c.memo_miss != 0
        {
            return true;
        }
    }
    false
}

fn fmt_dur_ms(d: Duration) -> String {
    let ms = d.as_secs_f64() * 1000.0;
    if ms < 1.0 {
        format!("{:.3} ms", ms)
    } else {
        format!("{:.1} ms", ms)
    }
}

fn show_profile_table(ui: &mut egui::Ui, p: &ProfileTotals) {
    ui.monospace(format!(
        "I/O 書き込み合計: {}",
        fmt_dur_ms(p.io_write_total)
    ));
    ui.add_space(4.0);
    egui::Grid::new("profile-grid")
        .striped(true)
        .num_columns(16)
        .show(ui, |ui| {
            ui.monospace("深さ");
            ui.monospace("nodes");
            ui.monospace("cand");
            ui.monospace("pruned");
            ui.monospace("leaves");
            ui.monospace("pre_thres");
            ui.monospace("pre_e1ng");
            ui.monospace("L-hit");
            ui.monospace("G-hit");
            ui.monospace("Miss");
            ui.monospace("gen");
            ui.monospace("assign");
            ui.monospace("upper");
            ui.monospace("fall");
            ui.monospace("hash");
            ui.monospace("memo_get/miss_compute/out");
            ui.end_row();

            for d in 0..=W {
                let c = p.dfs_counts[d];
                let t = p.dfs_times[d];
                ui.monospace(format!("{:>2}", d));
                ui.monospace(format!("{}", c.nodes));
                ui.monospace(format!("{}", c.cand_generated));
                ui.monospace(format!("{}", c.pruned_upper));
                ui.monospace(format!("{}", c.leaves));
                ui.monospace(format!("{}", c.leaf_pre_tshort));
                ui.monospace(format!("{}", c.leaf_pre_e1_impossible));
                ui.monospace(format!("{}", c.memo_lhit));
                ui.monospace(format!("{}", c.memo_ghit));
                ui.monospace(format!("{}", c.memo_miss));
                ui.monospace(fmt_dur_ms(t.gen_candidates));
                ui.monospace(fmt_dur_ms(t.assign_cols));
                ui.monospace(fmt_dur_ms(t.upper_bound));
                ui.monospace(fmt_dur_ms(t.leaf_fall_pre));
                ui.monospace(fmt_dur_ms(t.leaf_hash));
                ui.monospace(format!(
                    "{} / {} / {}",
                    fmt_dur_ms(t.leaf_memo_get),
                    fmt_dur_ms(t.leaf_memo_miss_compute),
                    fmt_dur_ms(t.out_serialize),
                ));
                ui.end_row();
            }
        });
}

#[derive(Clone, Copy)]
enum TCell {
    Blank,
    Any,
    Any4,
    Fixed(u8),
}

impl Cell {
    fn label_char(self) -> char {
        match self {
            Cell::Blank => '.',
            Cell::Any => 'N',
            Cell::Any4 => 'X',
            Cell::Abs(i) => (b'A' + i) as char,
            Cell::Fixed(c) => (b'0' + c) as char,
        }
    }
}

fn cell_style(c: Cell) -> (String, Color32, egui::Stroke) {
    match c {
        Cell::Blank => (
            "・".to_string(),
            Color32::WHITE,
            egui::Stroke::new(1.0, Color32::LIGHT_GRAY),
        ),
        Cell::Any => (
            "N".to_string(),
            Color32::from_rgb(254, 243, 199),
            egui::Stroke::new(1.0, Color32::from_rgb(245, 158, 11)),
        ),
        Cell::Any4 => (
            "X".to_string(),
            Color32::from_rgb(220, 252, 231),
            egui::Stroke::new(1.0, Color32::from_rgb(22, 163, 74)),
        ),
        Cell::Abs(i) => {
            let ch = (b'A' + i) as char;
            (
                ch.to_string(),
                Color32::from_rgb(238, 242, 255),
                egui::Stroke::new(1.0, Color32::from_rgb(99, 102, 241)),
            )
        }
        Cell::Fixed(i) => {
            // 0:R, 1:G, 2:B, 3:Y（表示は R/G/B/Y）
            match i {
                0 => (
                    "R".to_string(),
                    Color32::from_rgb(254, 226, 226),
                    egui::Stroke::new(1.0, Color32::from_rgb(239, 68, 68)),
                ),
                1 => (
                    "G".to_string(),
                    Color32::from_rgb(220, 252, 231),
                    egui::Stroke::new(1.0, Color32::from_rgb(34, 197, 94)),
                ),
                2 => (
                    "B".to_string(),
                    Color32::from_rgb(219, 234, 254),
                    egui::Stroke::new(1.0, Color32::from_rgb(59, 130, 246)),
                ),
                3 => (
                    "Y".to_string(),
                    Color32::from_rgb(254, 249, 195),
                    egui::Stroke::new(1.0, Color32::from_rgb(234, 179, 8)),
                ),
                _ => (
                    "?".to_string(),
                    Color32::LIGHT_GRAY,
                    egui::Stroke::new(1.0, Color32::DARK_GRAY),
                ),
            }
        }
    }
}
fn cycle_abs(c: Cell) -> Cell {
    match c {
        Cell::Blank | Cell::Any | Cell::Any4 => Cell::Abs(0),
        Cell::Abs(i) => Cell::Abs(((i as usize + 1) % 13) as u8),
        Cell::Fixed(_) => Cell::Abs(0),
    }
}
fn cycle_any(c: Cell) -> Cell {
    match c {
        Cell::Any => Cell::Any4,
        Cell::Any4 => Cell::Any,
        _ => Cell::Any,
    }
}
fn cycle_fixed(c: Cell) -> Cell {
    match c {
        Cell::Fixed(v) => Cell::Fixed(((v as usize + 1) % 4) as u8),
        _ => Cell::Fixed(0),
    }
}

fn draw_preview(ui: &mut egui::Ui, cols: &[[u16; W]; 4]) {
    let cell = 16.0_f32; // 1マスのサイズ
    let gap = 1.0_f32; // マス間の隙間

    let width = W as f32 * cell + (W - 1) as f32 * gap;
    let height = H as f32 * cell + (H - 1) as f32 * gap;

    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    // 0=R, 1=G, 2=B, 3=Y
    let palette = [
        Color32::from_rgb(239, 68, 68),  // red
        Color32::from_rgb(34, 197, 94),  // green
        Color32::from_rgb(59, 130, 246), // blue
        Color32::from_rgb(234, 179, 8),  // yellow
    ];

    for y in 0..H {
        for x in 0..W {
            let mut cidx: Option<usize> = None;
            let bit = 1u16 << y;
            if cols[0][x] & bit != 0 {
                cidx = Some(0);
            } else if cols[1][x] & bit != 0 {
                cidx = Some(1);
            } else if cols[2][x] & bit != 0 {
                cidx = Some(2);
            } else if cols[3][x] & bit != 0 {
                cidx = Some(3);
            }

            let fill = cidx.map(|k| palette[k]).unwrap_or(Color32::WHITE);

            let x0 = rect.min.x + x as f32 * (cell + gap);
            let y0 = rect.max.y - ((y + 1) as f32 * cell + y as f32 * gap);
            let r = egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(cell, cell));
            painter.rect_filled(r, 3.0, fill);
        }
    }
}

// ======== ここから：ビットボード最適化版 ========

// 6×14=84 マスを u128 にパック
type BB = u128;
const COL_BITS: usize = H; // 14

// 各端マスク（列境界/上下境界の越境を防ぐ）
const fn top_mask() -> BB {
    let mut m: BB = 0;
    let mut x = 0;
    while x < W {
        m |= 1u128 << (x * COL_BITS + (COL_BITS - 1));
        x += 1;
    }
    m
}
const fn bottom_mask() -> BB {
    let mut m: BB = 0;
    let mut x = 0;
    while x < W {
        m |= 1u128 << (x * COL_BITS);
        x += 1;
    }
    m
}
const TOP_MASK: BB = top_mask();
const BOTTOM_MASK: BB = bottom_mask();
const LEFTCOL_MASK: BB = (1u128 << COL_BITS) - 1;
const RIGHTCOL_MASK: BB = ((1u128 << COL_BITS) - 1) << ((W - 1) * COL_BITS);

// ★ 盤面全体の有効マスク
const fn board_mask() -> BB {
    let mut m: BB = 0;
    let mut x = 0;
    while x < W {
        m |= ((1u128 << COL_BITS) - 1) << (x * COL_BITS);
        x += 1;
    }
    m
}
const BOARD_MASK: BB = board_mask();

#[inline(always)]
fn pack_cols(cols: &[[u16; W]; 4]) -> [BB; 4] {
    let mut out = [0u128; 4];
    for c in 0..4 {
        let mut acc: BB = 0;
        for x in 0..W {
            acc |= (cols[c][x] as BB) << (x * COL_BITS);
        }
        out[c] = acc;
    }
    out
}

#[inline(always)]
fn unpack_mask_to_cols(mask: BB) -> [u16; W] {
    let mut out = [0u16; W];
    for (x, o) in out.iter_mut().enumerate() {
        *o = ((mask >> (x * COL_BITS)) as u16) & MASK14;
    }
    out
}

#[inline(always)]
fn neighbors(bits: BB) -> BB {
    let v_up = (bits & !TOP_MASK) << 1;
    let v_down = (bits & !BOTTOM_MASK) >> 1;
    let h_left = (bits & !LEFTCOL_MASK) >> COL_BITS;
    let h_right = (bits & !RIGHTCOL_MASK) << COL_BITS;
    v_up | v_down | h_left | h_right
}

// ======== 以降：探索ロジック（CLI版相当） ========

// 入力（A..M/N/X/./0..3）を元に抽象情報
struct AbstractInfo {
    labels: Vec<char>,
    adj: Vec<Vec<usize>>,
}

fn build_abstract_info(board: &[char]) -> AbstractInfo {
    let mut labels = Vec::new();
    for &v in board {
        if ('A'..='M').contains(&v) && !labels.contains(&v) {
            labels.push(v);
        }
    }
    let mut label_idx = HashMap::new();
    for (i, &c) in labels.iter().enumerate() {
        label_idx.insert(c, i);
    }
    let n = labels.len();
    let mut adj = vec![Vec::<usize>::new(); n];
    let dirs = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];
    for x in 0..W {
        for y in 0..H {
            let v = board[y * W + x];
            if !('A'..='M').contains(&v) {
                continue;
            }
            let id = label_idx[&v];
            for (dx, dy) in dirs {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx < 0 || nx >= W as isize || ny < 0 || ny >= H as isize {
                    continue;
                }
                let w = board[(ny as usize) * W + (nx as usize)];
                if ('A'..='M').contains(&w) && w != v {
                    let nb = label_idx[&w];
                    if !adj[id].contains(&nb) {
                        adj[id].push(nb);
                    }
                }
            }
        }
    }
    AbstractInfo { labels, adj }
}

// 4) 彩色: DSATUR 風で高速化（ビット集合）
fn enumerate_colorings_fast(info: &AbstractInfo) -> Vec<Vec<u8>> {
    let n = info.labels.len();
    if n == 0 {
        return vec![Vec::new()];
    }

    // 隣接を bitset に
    let mut adj = vec![0u16; n];
    for (v, adjv) in adj.iter_mut().enumerate() {
        let mut m = 0u16;
        for &u in &info.adj[v] {
            m |= 1u16 << u;
        }
        *adjv = m;
    }

    // DSATUR: 彩色飽和度最大→次数最大
    let mut color = vec![4u8; n]; // 0..=3, 4=未彩色
    let mut used_mask = vec![0u8; n]; // 4bit: 近傍で使われた色

    let mut out = Vec::new();
    fn dfs(
        vleft: usize,
        total_n: usize,
        adj: &[u16],
        color: &mut [u8],
        used_mask: &mut [u8],
        out: &mut Vec<Vec<u8>>,
        max_used: u8, // 既に使われた最大色（0始まり）。新規色は max_used+1 のみ許可
    ) {
        if vleft == 0 {
            out.push(color.to_vec());
            return;
        }

        // 次に塗る頂点を選択（DSATUR）
        let mut pick = None;
        let mut best_sat = -1i32;
        let mut best_deg = -1i32;
        for v in 0..color.len() {
            if color[v] != 4 {
                continue;
            }
            let sat = used_mask[v].count_ones() as i32;
            let deg = adj[v].count_ones() as i32;
            if sat > best_sat || (sat == best_sat && deg > best_deg) {
                best_sat = sat;
                best_deg = deg;
                pick = Some(v);
            }
        }
        let v = pick.unwrap();

        // 使える色を列挙（4色から used を除く）+ 対称性破り
        let forbid = used_mask[v];
        let mut new_color_limit = (max_used + 1).min(3);
        if vleft == total_n {
            new_color_limit = 0;
        } // 最初の1手は 0 のみ
        for c in 0u8..=new_color_limit {
            if ((forbid >> c) & 1) != 0 {
                continue;
            }
            color[v] = c;

            // 近傍の used_mask を更新
            let mut touched = 0u16;
            let mut nb = adj[v];
            while nb != 0 {
                let u = nb.trailing_zeros() as usize;
                nb &= nb - 1;
                if color[u] == 4 {
                    used_mask[u] |= 1u8 << c;
                    touched |= 1u16 << u;
                }
            }
            let next_max_used = if c > max_used { c } else { max_used };
            dfs(
                vleft - 1,
                total_n,
                adj,
                color,
                used_mask,
                out,
                next_max_used,
            );

            // ロールバック
            color[v] = 4;
            let mut t = touched;
            while t != 0 {
                let u = t.trailing_zeros() as usize;
                t &= t - 1;
                used_mask[u] &= !(1u8 << c);
            }
        }
    }
    dfs(n, n, &adj, &mut color, &mut used_mask, &mut out, 0);
    out
}

fn apply_coloring_to_template(base: &[char], map: &HashMap<char, u8>) -> Vec<TCell> {
    base.iter()
        .map(|&v| {
            if ('A'..='M').contains(&v) {
                TCell::Fixed(map[&v])
            } else if v == 'N' {
                TCell::Any
            } else if v == 'X' {
                TCell::Any4
            } else if v == '.' {
                TCell::Blank
            } else if ('0'..='3').contains(&v) {
                TCell::Fixed(v as u8 - b'0')
            } else {
                TCell::Blank
            }
        })
        .collect()
}

// 列DP（通り数）
fn count_column_candidates_dp(col: &[TCell]) -> BigUint {
    let mut dp0 = BigUint::one(); // belowBlank=false
    let mut dp1 = BigUint::zero(); // belowBlank=true
    for &cell in col.iter().take(H) {
        let mut ndp0 = BigUint::zero();
        let mut ndp1 = BigUint::zero();
        match cell {
            TCell::Blank => {
                ndp1 += &dp0;
                ndp1 += &dp1;
            }
            TCell::Any4 => {
                if !dp0.is_zero() {
                    ndp0 += dp0.clone() * BigUint::from(4u32);
                }
            }
            TCell::Any => {
                ndp1 += &dp0;
                ndp1 += &dp1;
                if !dp0.is_zero() {
                    ndp0 += dp0 * BigUint::from(4u32);
                }
            }
            TCell::Fixed(_) => {
                ndp0 += &dp0;
            }
        }
        dp0 = ndp0;
        dp1 = ndp1;
    }
    dp0 + dp1
}

// 列ストリーミング列挙（従来版）
fn stream_column_candidates<F: FnMut([u16; 4])>(col: &[TCell], mut yield_masks: F) {
    fn rec<F: FnMut([u16; 4])>(
        y: usize,
        below_blank: bool,
        col: &[TCell],
        masks: &mut [u16; 4],
        yield_masks: &mut F,
    ) {
        if y >= H {
            yield_masks(*masks);
            return;
        }
        match col[y] {
            TCell::Blank => rec(y + 1, true, col, masks, yield_masks),
            TCell::Any4 => {
                if !below_blank {
                    for c in 0..4 {
                        masks[c] |= 1 << y;
                        rec(y + 1, false, col, masks, yield_masks);
                        masks[c] &= !(1 << y);
                    }
                }
            }
            TCell::Any => {
                rec(y + 1, true, col, masks, yield_masks);
                if !below_blank {
                    for c in 0..4 {
                        masks[c] |= 1 << y;
                        rec(y + 1, false, col, masks, yield_masks);
                        masks[c] &= !(1 << y);
                    }
                }
            }
            TCell::Fixed(c) => {
                if !below_blank {
                    masks[c as usize] |= 1 << y;
                    rec(y + 1, false, col, masks, yield_masks);
                    masks[c as usize] &= !(1 << y);
                }
            }
        }
    }
    let mut masks = [0u16; 4];
    rec(0, false, col, &mut masks, &mut yield_masks);
}

// 列ストリーミング列挙（計測版：列挙時間＝再帰本体、yield 時間は除外）
fn stream_column_candidates_timed<F: FnMut([u16; 4])>(
    col: &[TCell],
    enum_time: &mut Duration,
    mut yield_masks: F,
) {
    fn rec<F: FnMut([u16; 4])>(
        y: usize,
        below_blank: bool,
        col: &[TCell],
        masks: &mut [u16; 4],
        enum_time: &mut Duration,
        last_start: &mut Instant,
        yield_masks: &mut F,
    ) {
        if y >= H {
            *enum_time += last_start.elapsed();
            yield_masks(*masks);
            *last_start = Instant::now();
            return;
        }
        match col[y] {
            TCell::Blank => rec(y + 1, true, col, masks, enum_time, last_start, yield_masks),
            TCell::Any4 => {
                if !below_blank {
                    for c in 0..4 {
                        masks[c] |= 1 << y;
                        rec(y + 1, false, col, masks, enum_time, last_start, yield_masks);
                        masks[c] &= !(1 << y);
                    }
                }
            }
            TCell::Any => {
                rec(y + 1, true, col, masks, enum_time, last_start, yield_masks);
                if !below_blank {
                    for c in 0..4 {
                        masks[c] |= 1 << y;
                        rec(y + 1, false, col, masks, enum_time, last_start, yield_masks);
                        masks[c] &= !(1 << y);
                    }
                }
            }
            TCell::Fixed(c) => {
                if !below_blank {
                    masks[c as usize] |= 1 << y;
                    rec(y + 1, false, col, masks, enum_time, last_start, yield_masks);
                    masks[c as usize] &= !(1 << y);
                }
            }
        }
    }
    let mut masks = [0u16; 4];
    let mut last_start = Instant::now();
    rec(
        0,
        false,
        col,
        &mut masks,
        enum_time,
        &mut last_start,
        &mut yield_masks,
    );
    *enum_time += last_start.elapsed();
}

// 5) 列候補生成のハイブリッド（小さい列だけ前展開）
enum ColGen {
    Pre(Vec<[u16; 4]>),
    Stream(Vec<TCell>),
}
fn build_colgen(col: &[TCell], cnt: &BigUint) -> ColGen {
    if cnt.bits() <= 11 {
        let mut v = Vec::new();
        stream_column_candidates(col, |m| v.push(m));
        ColGen::Pre(v)
    } else {
        ColGen::Stream(col.to_vec())
    }
}

// ====== 落下：スカラ版と PEXT/PDEP 版 ======
#[inline(always)]
fn fall_cols(cols_in: &[[u16; W]; 4]) -> [[u16; W]; 4] {
    let mut out = [[0u16; W]; 4];

    for x in 0..W {
        let c0 = cols_in[0][x] & MASK14;
        let c1 = cols_in[1][x] & MASK14;
        let c2 = cols_in[2][x] & MASK14;
        let c3 = cols_in[3][x] & MASK14;
        let mut occ = c0 | c1 | c2 | c3;

        let mut dst: usize = 0;
        while occ != 0 {
            let bit = occ & occ.wrapping_neg();
            let color = if (c0 & bit) != 0 {
                0
            } else if (c1 & bit) != 0 {
                1
            } else if (c2 & bit) != 0 {
                2
            } else {
                3
            };
            out[color][x] |= 1u16 << dst;
            dst += 1;
            occ &= occ - 1;
        }
    }

    out
}

#[inline(always)]
fn fall_cols_fast(cols_in: &[[u16; W]; 4]) -> [[u16; W]; 4] {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("bmi2") {
            unsafe {
                return fall_cols_bmi2(cols_in);
            }
        }
    }
    fall_cols(cols_in)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "bmi2")]
unsafe fn fall_cols_bmi2(cols_in: &[[u16; W]; 4]) -> [[u16; W]; 4] {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{_pdep_u32, _pext_u32};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{_pdep_u32, _pext_u32};

    let mut out = [[0u16; W]; 4];

    for x in 0..W {
        let c0 = cols_in[0][x] as u32;
        let c1 = cols_in[1][x] as u32;
        let c2 = cols_in[2][x] as u32;
        let c3 = cols_in[3][x] as u32;

        let occ = (c0 | c1 | c2 | c3) & (MASK14 as u32);
        let k = occ.count_ones();
        if k == 0 {
            continue;
        }
        let base = (1u32 << k) - 1;

        let s0 = _pext_u32(c0, occ);
        let s1 = _pext_u32(c1, occ);
        let s2 = _pext_u32(c2, occ);
        let s3 = _pext_u32(c3, occ);

        out[0][x] = _pdep_u32(s0, base) as u16;
        out[1][x] = _pdep_u32(s1, base) as u16;
        out[2][x] = _pdep_u32(s2, base) as u16;
        out[3][x] = _pdep_u32(s3, base) as u16;
    }

    out
}

// 連結抽出（旧ビット列版：他箇所でも使用しているので残す）
#[allow(dead_code)]
#[inline(always)]
fn component_from_seed_cols(s: &[u16; W], seed_x: usize, seed_bits: u16) -> [u16; W] {
    let mut comp = [0u16; W];
    let mut frontier = [0u16; W];
    comp[seed_x] = seed_bits;
    frontier[seed_x] = seed_bits;
    loop {
        let mut changed = false;
        let mut next = [0u16; W];
        for x in 0..W {
            let mut nb = ((frontier[x] << 1) & MASK14) | (frontier[x] >> 1);
            if x > 0 {
                nb |= frontier[x - 1];
            }
            if x + 1 < W {
                nb |= frontier[x + 1];
            }
            next[x] = nb & s[x];
        }
        for x in 0..W {
            let add = next[x] & !comp[x];
            if add != 0 {
                comp[x] |= add;
                frontier[x] = add;
                changed = true;
            } else {
                frontier[x] = 0;
            }
        }
        if !changed {
            break;
        }
    }
    comp
}

#[inline(always)]
fn compute_erase_mask_cols(cols: &[[u16; W]; 4]) -> [u16; W] {
    let bb = pack_cols(cols);
    let mut clear_all: BB = 0;

    for &mask in bb.iter() {
        if mask.count_ones() < 4 {
            continue;
        }

        let mut s = mask;
        while s != 0 {
            let seed = s & (!s + 1);
            let mut comp = seed;
            let mut frontier = seed;
            loop {
                let grow = neighbors(frontier) & mask & !comp;
                if grow == 0 {
                    break;
                }
                comp |= grow;
                frontier = grow;
            }
            let sz = comp.count_ones();
            if sz >= 4 {
                clear_all |= comp;
            }
            s &= !comp;
        }
    }

    unpack_mask_to_cols(clear_all)
}

// ★ 4個消しモード用：マスク + “5個以上あったか” + “4個があったか”
#[inline(always)]
fn compute_erase_mask_and_flags(cols: &[[u16; W]; 4]) -> ([u16; W], bool, bool) {
    let bb = pack_cols(cols);
    let mut clear_all: BB = 0;
    let mut had_ge5 = false;
    let mut had_four = false;

    for &mask in bb.iter() {
        if mask.count_ones() < 4 {
            continue;
        }
        let mut s = mask;
        while s != 0 {
            let seed = s & (!s + 1);
            let mut comp = seed;
            let mut frontier = seed;
            loop {
                let grow = neighbors(frontier) & mask & !comp;
                if grow == 0 {
                    break;
                }
                comp |= grow;
                frontier = grow;
            }
            let sz = comp.count_ones();
            if sz >= 4 {
                clear_all |= comp;
                if sz == 4 {
                    had_four = true;
                } else {
                    had_ge5 = true;
                }
            }
            s &= !comp;
        }
    }

    (unpack_mask_to_cols(clear_all), had_ge5, had_four)
}

#[inline(always)]
fn apply_given_clear_and_fall(pre: &[[u16; W]; 4], clear: &[u16; W]) -> [[u16; W]; 4] {
    let mut next = [[0u16; W]; 4];

    let mut keep = [0u16; W];
    for (x, k) in keep.iter_mut().enumerate() {
        *k = (!clear[x]) & MASK14;
    }
    for (next_col, pre_col) in next.iter_mut().zip(pre.iter()) {
        for (&k, (cell, &p)) in keep.iter().zip(next_col.iter_mut().zip(pre_col.iter())) {
            *cell = p & k;
        }
    }
    fall_cols_fast(&next)
}

// ★ 追加：どの色か一色でも“4つ以上”あれば次の消去が起こりうる
#[inline(always)]
fn any_color_has_four(cols: &[[u16; W]; 4]) -> bool {
    let bb = pack_cols(cols);
    (bb[0].count_ones() >= 4)
        || (bb[1].count_ones() >= 4)
        || (bb[2].count_ones() >= 4)
        || (bb[3].count_ones() >= 4)
}

#[inline(always)]
fn apply_erase_and_fall_cols(cols: &[[u16; W]; 4]) -> (bool, [[u16; W]; 4]) {
    if !any_color_has_four(cols) {
        return (false, *cols);
    }

    let clear = compute_erase_mask_cols(cols);
    let any = (0..W).any(|x| clear[x] != 0);
    if !any {
        (false, *cols)
    } else {
        (true, apply_given_clear_and_fall(cols, &clear))
    }
}

// ★ 4個消しモード用
enum StepExact {
    NoClear,
    Cleared([[u16; W]; 4]),
    Illegal,
}

#[inline(always)]
fn apply_erase_and_fall_exact4(cols: &[[u16; W]; 4]) -> StepExact {
    if !any_color_has_four(cols) {
        return StepExact::NoClear;
    }
    let (clear, had_ge5, had_four) = compute_erase_mask_and_flags(cols);
    if had_ge5 {
        return StepExact::Illegal;
    }
    let any = (0..W).any(|x| clear[x] != 0);
    if !any || !had_four {
        StepExact::NoClear
    } else {
        StepExact::Cleared(apply_given_clear_and_fall(cols, &clear))
    }
}

// 列 x への 4色一括代入（ループ展開）
#[inline(always)]
fn assign_col_unrolled(cols: &mut [[u16; W]; 4], x: usize, masks: &[u16; 4]) {
    // 安全のためのデバッグアサート（最適化時は消える）
    debug_assert!(x < W);
    cols[0][x] = masks[0];
    cols[1][x] = masks[1];
    cols[2][x] = masks[2];
    cols[3][x] = masks[3];
}

// 列 x をゼロクリア（ループ展開）
#[inline(always)]
fn clear_col_unrolled(cols: &mut [[u16; W]; 4], x: usize) {
    debug_assert!(x < W);
    cols[0][x] = 0;
    cols[1][x] = 0;
    cols[2][x] = 0;
    cols[3][x] = 0;
}

// E1単一連結 + 追加条件 + T到達（最適化版）
#[inline(always)]
fn reaches_t_from_pre_single_e1(pre: &[[u16; W]; 4], t: u32, exact_four_only: bool) -> bool {
    if exact_four_only {
        let mut potential: u32 = 0;
        for col in pre.iter() {
            let cnt: u32 = col.iter().map(|&m| m.count_ones()).sum();
            potential = potential.saturating_add(cnt / 4);
        }
        if potential < t {
            return false;
        }
    }

    let bb_pre = pack_cols(pre);
    let (clear_bb, total_cnt) = {
        let mut clr: BB = 0;
        let mut tot: u32 = 0;
        for &bb in bb_pre.iter() {
            if bb.count_ones() < 4 {
                continue;
            }
            let mut s = bb;
            while s != 0 {
                let seed = s & (!s + 1);
                let mut comp = seed;
                let mut frontier = seed;
                loop {
                    let grow = neighbors(frontier) & bb & !comp;
                    if grow == 0 {
                        break;
                    }
                    comp |= grow;
                    frontier = grow;
                }
                let sz = comp.count_ones();
                if sz >= 4 {
                    clr |= comp;
                    tot = tot.saturating_add(sz);
                }
                s &= !comp;
            }
        }
        (clr, tot)
    };

    let total = total_cnt;
    if total == 0 {
        return false;
    }

    // exact4 の場合、初回消去が 4 以外なら即不成立（以降の高コスト判定を回避）
    if exact_four_only && total != 4 {
        return false;
    }

    // 先に空白隣接とオーバーハングの簡易チェックで早期棄却
    let occ_bb = bb_pre[0] | bb_pre[1] | bb_pre[2] | bb_pre[3];
    let blank_bb = BOARD_MASK & !occ_bb;
    if neighbors(clear_bb) & blank_bb == 0 {
        return false;
    }

    let mut ok_overhang = false;
    for x in 0..W {
        let clear_col: u16 = ((clear_bb >> (x * COL_BITS)) as u16) & MASK14;
        if clear_col == 0 {
            continue;
        }
        let occ_col: u16 = ((occ_bb >> (x * COL_BITS)) as u16) & MASK14;

        let top_y = 15 - clear_col.leading_zeros() as usize;

        let above = (occ_col & !clear_col) >> (top_y + 1);
        let run = (above.trailing_ones()) as usize;

        if run <= 1 {
            ok_overhang = true;
            break;
        }
    }
    if !ok_overhang {
        return false;
    }

    // E1 単一連結チェック（clear_bb が 1 コンポーネントか）
    let seed = clear_bb & (!clear_bb + 1);
    let mut comp = seed;
    let mut frontier = seed;
    loop {
        let grow = neighbors(frontier) & clear_bb & !comp;
        if grow == 0 {
            break;
        }
        comp |= grow;
        frontier = grow;
    }
    if comp.count_ones() != total {
        return false;
    }

    let mut cur;
    {
        let mut work = [[0u16; W]; 4];
        let clear_cols = unpack_mask_to_cols(clear_bb);
        for x in 0..W {
            let inv = (!clear_cols[x]) & MASK14;
            work[0][x] = pre[0][x] & inv;
            work[1][x] = pre[1][x] & inv;
            work[2][x] = pre[2][x] & inv;
            work[3][x] = pre[3][x] & inv;
        }
        cur = fall_cols_fast(&work);
    }

    if t == 1 {
        return true;
    }

    // 残り (t-1) 連鎖のポテンシャル上限チェック
    {
        let mut potential: u32 = 0;
        for col in cur.iter() {
            let cnt: u32 = col.iter().map(|&m| m.count_ones()).sum();
            potential = potential.saturating_add(cnt / 4);
        }
        if potential < (t - 1) {
            return false;
        }
    }

    if !exact_four_only {
        for _ in 2..=t {
            let (erased, next) = apply_erase_and_fall_cols(&cur);
            if !erased {
                return false;
            }
            cur = next;
        }
        true
    } else {
        for _ in 2..=t {
            match apply_erase_and_fall_exact4(&cur) {
                StepExact::Illegal => return false,
                StepExact::NoClear => return false,
                StepExact::Cleared(next) => {
                    cur = next;
                }
            }
        }
        true
    }
}

// ========== 追撃最適化：占有比較の u128 化 & ハッシュの占有ビット走査 ==========

// 占有パターンの16bit整列パックを作って左右比較（u128 で一発）
// Some(false)=正方向, Some(true)=ミラー, None=完全同一（左右対称）
#[inline(always)]
fn choose_mirror_by_occupancy(cols: &[[u16; W]; 4]) -> Option<bool> {
    // 各列の占有（14bit）を 16bit チャンクに入れて 96bit（u128）に連結
    let mut packed: u128 = 0;
    let mut rev: u128 = 0;
    for x in 0..W {
        let occ = (cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) as u128;
        packed |= occ << (x * 16);
        rev |= occ << ((W - 1 - x) * 16);
    }
    if packed < rev {
        Some(false)
    } else if packed > rev {
        Some(true)
    } else {
        None
    }
}

// LUT 風：色マッピングの更新（未割当なら next を払い出し）
#[inline(always)]
fn map_code_lut(entry: &mut u8, next: &mut u8) -> u64 {
    if *entry == u8::MAX {
        *entry = *next;
        *next = next.wrapping_add(1);
    }
    *entry as u64
}

// ====== ここを差し替え：空白を P^k でまとめ掛けする高速版 ======
#[inline(always)]
fn canonical_hash64_oriented_bits(cols: &[[u16; W]; 4], mirror: bool) -> u64 {
    const P: u64 = 1099511628211;
    const O: u64 = 14695981039346656037;

    // P^k（k=0..14）をその場で構築（到達葉のみで実行されるため十分軽量）
    let mut p_pow = [1u64; 15];
    for i in 1..15 {
        p_pow[i] = p_pow[i - 1].wrapping_mul(P);
    }
    #[inline(always)]
    fn mul_pow(h: u64, pp: &[u64; 15], k: usize) -> u64 {
        debug_assert!(k < 15);
        h.wrapping_mul(pp[k])
    }

    let mut h = O;
    let mut map: [u8; 4] = [u8::MAX; 4];
    let mut next: u8 = 1;

    if !mirror {
        for xi in 0..W {
            let c0 = cols[0][xi];
            let c1 = cols[1][xi];
            let c2 = cols[2][xi];
            let c3 = cols[3][xi];
            let mut occ: u16 = (c0 | c1 | c2 | c3) as u16;

            if occ == 0 {
                // 列が空：空白14個ぶん一気に掛ける
                h = mul_pow(h, &p_pow, 14);
                continue;
            }

            // 直前の占有 y（最初は -1 とみなす）
            let mut prev_y: i32 = -1;

            while occ != 0 {
                let bit = occ & occ.wrapping_neg();
                let y = bit.trailing_zeros() as i32;

                // 占有間の空白をまとめて掛ける
                let gap = (y - prev_y - 1) as usize;
                h = mul_pow(h, &p_pow, gap);

                // 色コード（初出→1, 次→2...）
                let code = if (c0 & bit) != 0 {
                    map_code_lut(&mut map[0], &mut next)
                } else if (c1 & bit) != 0 {
                    map_code_lut(&mut map[1], &mut next)
                } else if (c2 & bit) != 0 {
                    map_code_lut(&mut map[2], &mut next)
                } else {
                    map_code_lut(&mut map[3], &mut next)
                };

                h ^= code;
                h = h.wrapping_mul(P);

                prev_y = y;
                occ &= occ - 1;
            }

            // 列末尾の空白をまとめて掛ける
            let tail = (14 - (prev_y as usize) - 1) as usize;
            h = mul_pow(h, &p_pow, tail);
        }
    } else {
        for xr in (0..W).rev() {
            let c0 = cols[0][xr];
            let c1 = cols[1][xr];
            let c2 = cols[2][xr];
            let c3 = cols[3][xr];
            let mut occ: u16 = (c0 | c1 | c2 | c3) as u16;

            if occ == 0 {
                h = mul_pow(h, &p_pow, 14);
                continue;
            }

            let mut prev_y: i32 = -1;

            while occ != 0 {
                let bit = occ & occ.wrapping_neg();
                let y = bit.trailing_zeros() as i32;

                let gap = (y - prev_y - 1) as usize;
                h = mul_pow(h, &p_pow, gap);

                let code = if (c0 & bit) != 0 {
                    map_code_lut(&mut map[0], &mut next)
                } else if (c1 & bit) != 0 {
                    map_code_lut(&mut map[1], &mut next)
                } else if (c2 & bit) != 0 {
                    map_code_lut(&mut map[2], &mut next)
                } else {
                    map_code_lut(&mut map[3], &mut next)
                };

                h ^= code;
                h = h.wrapping_mul(P);

                prev_y = y;
                occ &= occ - 1;
            }

            let tail = (14 - (prev_y as usize) - 1) as usize;
            h = mul_pow(h, &p_pow, tail);
        }
    }

    h
}

// 占有で決まらない（左右対称）場合のみ、両向き計算して小さい方を採用
#[inline(always)]
fn canonical_hash64_fast(cols: &[[u16; W]; 4]) -> (u64, bool) {
    if let Some(mirror) = choose_mirror_by_occupancy(cols) {
        let h = canonical_hash64_oriented_bits(cols, mirror);
        (h, mirror)
    } else {
        let h0 = canonical_hash64_oriented_bits(cols, false);
        let h1 = canonical_hash64_oriented_bits(cols, true);
        if h0 <= h1 {
            (h0, false)
        } else {
            (h1, true)
        }
    }
}

fn encode_canonical_string(cols: &[[u16; W]; 4], mirror: bool) -> String {
    let mut map: [u8; 4] = [0; 4];
    let mut next: u8 = b'A';
    let mut s = String::with_capacity(W * H);
    if !mirror {
        for x in 0..W {
            for y in 0..H {
                let bit = 1u16 << y;
                let color = if cols[0][x] & bit != 0 {
                    0
                } else if cols[1][x] & bit != 0 {
                    1
                } else if cols[2][x] & bit != 0 {
                    2
                } else if cols[3][x] & bit != 0 {
                    3
                } else {
                    4
                };
                if color == 4 {
                    s.push('.');
                } else {
                    if map[color] == 0 {
                        map[color] = next;
                        next = next.wrapping_add(1);
                    }
                    s.push(map[color] as char);
                }
            }
        }
    } else {
        for x in (0..W).rev() {
            for y in 0..H {
                let bit = 1u16 << y;
                let color = if cols[0][x] & bit != 0 {
                    0
                } else if cols[1][x] & bit != 0 {
                    1
                } else if cols[2][x] & bit != 0 {
                    2
                } else if cols[3][x] & bit != 0 {
                    3
                } else {
                    4
                };
                if color == 4 {
                    s.push('.');
                } else {
                    if map[color] == 0 {
                        map[color] = next;
                        next = next.wrapping_add(1);
                    }
                    s.push(map[color] as char);
                }
            }
        }
    }
    s
}
fn serialize_board_from_cols(cols: &[[u16; W]; 4]) -> Vec<String> {
    let mut rows = Vec::with_capacity(H);
    for y in 0..H {
        let mut line = String::with_capacity(W);
        for x in 0..W {
            let bit = 1u16 << y;
            let ch = if cols[0][x] & bit != 0 {
                '0'
            } else if cols[1][x] & bit != 0 {
                '1'
            } else if cols[2][x] & bit != 0 {
                '2'
            } else if cols[3][x] & bit != 0 {
                '3'
            } else {
                '.'
            };
            line.push(ch);
        }
        rows.push(line);
    }
    rows
}
fn fnv1a32(s: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for &b in s.as_bytes() {
        h ^= b as u32;
        h = h
            .wrapping_add(h << 1)
            .wrapping_add(h << 4)
            .wrapping_add(h << 7)
            .wrapping_add(h << 8)
            .wrapping_add(h << 24);
    }
    h
}

// 3) JSON を手組みで生成（serde_json を避ける）
#[inline(always)]
fn make_json_line_str(
    key: &str,
    hash: u32,
    chains: u32,
    rows: &[String],
    mapping: &HashMap<char, u8>,
    mirror: bool,
) -> String {
    let mut s = String::with_capacity(256);
    s.push('{');
    s.push_str(r#""key":"#);
    s.push('"');
    for ch in key.chars() {
        if ch == '"' {
            s.push('\\');
        }
        s.push(ch);
    }
    s.push('"');
    s.push(',');

    s.push_str(r#""hash":"#);
    let _ = std::fmt::write(&mut s, format_args!("{}", hash));
    s.push(',');
    s.push_str(r#""chains":"#);
    let _ = std::fmt::write(&mut s, format_args!("{}", chains));
    s.push(',');

    s.push_str(r#""pre_chain_board":["#);
    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('"');
        for ch in row.chars() {
            if ch == '"' {
                s.push('\\');
            }
            s.push(ch);
        }
        s.push('"');
    }
    s.push_str("],");

    s.push_str(r#""example_mapping":{"#);
    let mut keys: Vec<_> = mapping.keys().copied().collect();
    keys.sort_unstable();
    let mut first = true;
    for k in keys {
        if !first {
            s.push(',');
        }
        first = false;
        s.push('"');
        s.push(k);
        s.push_str(r#"":"#);
        let _ = std::fmt::write(&mut s, format_args!("{}", mapping[&k]));
    }
    s.push_str("},");

    s.push_str(r#""mirror":"#);
    s.push_str(if mirror { "true" } else { "false" });
    s.push('}');
    s
}

// 近似LRU（ローカル専用）
struct ApproxLru {
    limit: usize,
    map: U64Map<bool>, // 使わない（今回の方針では未参照でもOK）
    q: VecDeque<u64>,
}
impl ApproxLru {
    fn new(limit: usize) -> Self {
        let cap = (limit.saturating_mul(11) / 10).max(16);
        let map: U64Map<bool> =
            std::collections::HashMap::with_capacity_and_hasher(cap, BuildNoHashHasher::default());
        let q = VecDeque::with_capacity(cap);
        Self { limit, map, q }
    }
    #[allow(dead_code)]
    fn get(&self, k: u64) -> Option<bool> {
        self.map.get(&k).copied()
    }
    #[allow(dead_code)]
    fn insert(&mut self, k: u64, v: bool) {
        use std::collections::hash_map::Entry;
        match self.map.entry(k) {
            Entry::Vacant(e) => {
                e.insert(v);
                self.q.push_back(k);
                let cap = (self.limit as f64 * 1.1) as usize;
                if self.q.len() > cap {
                    let to_delete = self.q.len() - self.limit;
                    for _ in 0..to_delete {
                        if let Some(kk) = self.q.pop_front() {
                            self.map.remove(&kk);
                        }
                    }
                }
            }
            Entry::Occupied(mut e) => {
                e.insert(v);
            }
        }
    }
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.map.len()
    }
}

// DFS（列順最適化、ハイブリッド列候補、ローカルLRU、グローバル一意集合、バッチ出力）
#[allow(clippy::too_many_arguments)]
fn dfs_combine_parallel(
    depth: usize,
    cols0: &mut [[u16; W]; 4],
    gens: &[ColGen; W],
    order: &[usize],
    threshold: u32,
    exact_four_only: bool,
    _memo: &mut ApproxLru, // 現方針では実質未使用（陽性のみ挿入なら使用可）
    local_output_once: &mut U64Set,
    global_output_once: &Arc<DU64Set>,
    _global_memo: &Arc<DU64Map<bool>>, // get を廃止
    map_label_to_color: &HashMap<char, u8>,
    batch: &mut Vec<String>,
    batch_sender: &Sender<Vec<String>>,
    stat_sender: &Sender<StatDelta>,
    // 計測
    profile_enabled: bool,
    time_batch: &mut TimeDelta,

    nodes_batch: &mut u64,
    leaves_batch: &mut u64,
    outputs_batch: &mut u64,
    pruned_batch: &mut u64,
    lhit_batch: &mut u64,
    ghit_batch: &mut u64,
    mmiss_batch: &mut u64,
    preview_ok: bool,
    preview_tx: &Sender<Message>,
    last_preview: &mut Instant,
    _lru_limit: usize,
    t0: Instant,
    abort: &AtomicBool,
    placed_total: u32,
    remain_suffix: &[u16],
) -> Result<()> {
    if abort.load(Ordering::Relaxed) {
        return Ok(());
    }
    *nodes_batch += 1;
    if profile_enabled {
        time_batch.dfs_counts[depth].nodes += 1;
    }

    // ==== 葉はここで処理（上界チェックはスキップ！）====
    if depth == W {
        *leaves_batch += 1;
        if profile_enabled {
            time_batch.dfs_counts[depth].leaves += 1;
        }

        // ★ 早期リターン（落下や到達判定より前）
        if placed_total < 4 * threshold {
            // 4T 未満はどう頑張っても T 連鎖に届かない
            if profile_enabled {
                time_batch.dfs_counts[depth].leaf_pre_tshort += 1;
            }
            return Ok(());
        }
        if !any_color_has_four(cols0) {
            // E1 不可能
            if profile_enabled {
                time_batch.dfs_counts[depth].leaf_pre_e1_impossible += 1;
            }
            return Ok(());
        }

        // ★ 初回 fall をスキップ（列生成で既に重力正規化済み）
        let pre = *cols0;

        // ==== 先に到達判定のみ実行（ミス計上＋計測は miss_compute に積む）====
        if profile_enabled {
            time_batch.dfs_counts[depth].memo_miss += 1;
        }
        *mmiss_batch += 1;
        let reached = prof!(
            profile_enabled,
            time_batch.dfs_times[depth].leaf_memo_miss_compute,
            { reaches_t_from_pre_single_e1(&pre, threshold, exact_four_only) }
        );
        if !reached {
            // 統計フラッシュ（従来通り）
            if (*nodes_batch >= 4096 || t0.elapsed().as_millis() % 500 == 0)
                && (*nodes_batch > 0
                    || *leaves_batch > 0
                    || *outputs_batch > 0
                    || *pruned_batch > 0
                    || *lhit_batch > 0
                    || *ghit_batch > 0
                    || *mmiss_batch > 0)
            {
                let _ = stat_sender.send(StatDelta {
                    nodes: *nodes_batch,
                    leaves: *leaves_batch,
                    outputs: *outputs_batch,
                    pruned: *pruned_batch,
                    lhit: *lhit_batch,
                    ghit: *ghit_batch,
                    mmiss: *mmiss_batch,
                });
                *nodes_batch = 0;
                *leaves_batch = 0;
                *outputs_batch = 0;
                *pruned_batch = 0;
                *lhit_batch = 0;
                *ghit_batch = 0;
                *mmiss_batch = 0;

                if profile_enabled && time_delta_has_any(time_batch) {
                    let td = time_batch.clone();
                    *time_batch = TimeDelta::default();
                    let _ = preview_tx.send(Message::TimeDelta(td));
                }
            }
            return Ok(());
        }

        // ==== 到達した場合にだけ正規化キー生成（計測は leaf_hash に積む）====
        let (key64, mirror) = prof!(profile_enabled, time_batch.dfs_times[depth].leaf_hash, {
            canonical_hash64_fast(&pre)
        });

        // （必要なら）陽性だけローカル LRU に入れる
        // if _lru_limit > 0 { _memo.insert(key64, true); }

        if !local_output_once.contains(&key64) {
            // 近似的なグローバル一意集合で重複回避
            const OUTPUT_SET_CAP: usize = 2_000_000;
            const OUTPUT_SAMPLE_MASK: u64 = 0x7; // 1/8 サンプリング
            const OUTPUT_SAMPLE_MATCH: u64 = 0;

            let under_cap = global_output_once.len() < OUTPUT_SET_CAP;
            let sampled = (key64 & OUTPUT_SAMPLE_MASK) == OUTPUT_SAMPLE_MATCH;
            let should_insert_global = preview_ok && under_cap && sampled;

            let is_new = if should_insert_global {
                global_output_once.insert(key64)
            } else {
                !global_output_once.contains(&key64)
            };

            if is_new {
                local_output_once.insert(key64);

                if preview_ok && last_preview.elapsed() >= Duration::from_millis(3000) {
                    let _ = preview_tx.send(Message::Preview(pre));
                    *last_preview = Instant::now();
                }

                // 出力整形
                let line = prof!(
                    profile_enabled,
                    time_batch.dfs_times[depth].out_serialize,
                    {
                        let key_str = encode_canonical_string(&pre, mirror);
                        let hash = fnv1a32(&key_str);
                        let rows = serialize_board_from_cols(&pre);
                        make_json_line_str(
                            &key_str,
                            hash,
                            threshold,
                            &rows,
                            map_label_to_color,
                            mirror,
                        )
                    }
                );
                batch.push(line);
                *outputs_batch += 1;

                if batch.len() >= 2048 {
                    let out = std::mem::take(batch);
                    let _ = batch_sender.send(out);
                }
            }
        }

        // 統計フラッシュ
        if (*nodes_batch >= 4096 || t0.elapsed().as_millis() % 500 == 0)
            && (*nodes_batch > 0
                || *leaves_batch > 0
                || *outputs_batch > 0
                || *pruned_batch > 0
                || *lhit_batch > 0
                || *ghit_batch > 0
                || *mmiss_batch > 0)
        {
            let _ = stat_sender.send(StatDelta {
                nodes: *nodes_batch,
                leaves: *leaves_batch,
                outputs: *outputs_batch,
                pruned: *pruned_batch,
                lhit: *lhit_batch,
                ghit: *ghit_batch,
                mmiss: *mmiss_batch,
            });
            *nodes_batch = 0;
            *leaves_batch = 0;
            *outputs_batch = 0;
            *pruned_batch = 0;
            *lhit_batch = 0;
            *ghit_batch = 0;
            *mmiss_batch = 0;

            if profile_enabled && time_delta_has_any(time_batch) {
                let td = time_batch.clone();
                *time_batch = TimeDelta::default();
                let _ = preview_tx.send(Message::TimeDelta(td));
            }
        }
        return Ok(());
    }

    // ===== 上界枝刈り（非葉のみ）=====
    let pruned_now = prof!(profile_enabled, time_batch.dfs_times[depth].upper_bound, {
        let placed = placed_total;
        let remain_cap = *remain_suffix.get(depth).unwrap_or(&0) as u32;
        placed + remain_cap < 4 * threshold
    });
    if pruned_now {
        *pruned_batch += 1;
        if profile_enabled {
            time_batch.dfs_counts[depth].pruned_upper += 1;
        }
        if (*nodes_batch >= 4096 || t0.elapsed().as_millis() % 500 == 0)
            && (*nodes_batch > 0
                || *leaves_batch > 0
                || *outputs_batch > 0
                || *pruned_batch > 0
                || *lhit_batch > 0
                || *ghit_batch > 0
                || *mmiss_batch > 0)
        {
            let _ = stat_sender.send(StatDelta {
                nodes: *nodes_batch,
                leaves: *leaves_batch,
                outputs: *outputs_batch,
                pruned: *pruned_batch,
                lhit: *lhit_batch,
                ghit: *ghit_batch,
                mmiss: *mmiss_batch,
            });
            *nodes_batch = 0;
            *leaves_batch = 0;
            *outputs_batch = 0;
            *pruned_batch = 0;
            *lhit_batch = 0;
            *ghit_batch = 0;
            *mmiss_batch = 0;

            if profile_enabled && time_delta_has_any(time_batch) {
                let td = time_batch.clone();
                *time_batch = TimeDelta::default();
                let _ = preview_tx.send(Message::TimeDelta(td));
            }
        }
        return Ok(());
    }

    let x = order[depth];
    match &gens[x] {
        ColGen::Pre(v) => {
            if profile_enabled {
                time_batch.dfs_counts[depth].cand_generated += v.len() as u64;
            }
            for &masks in v {
                if abort.load(Ordering::Relaxed) {
                    return Ok(());
                }
                prof!(profile_enabled, time_batch.dfs_times[depth].assign_cols, {
                    assign_col_unrolled(cols0, x, &masks);
                });
                let add = (0..4).map(|c| masks[c].count_ones()).sum::<u32>();
                let _ = dfs_combine_parallel(
                    depth + 1,
                    cols0,
                    gens,
                    order,
                    threshold,
                    exact_four_only,
                    _memo,
                    local_output_once,
                    global_output_once,
                    _global_memo,
                    map_label_to_color,
                    batch,
                    batch_sender,
                    stat_sender,
                    profile_enabled,
                    time_batch,
                    nodes_batch,
                    leaves_batch,
                    outputs_batch,
                    pruned_batch,
                    lhit_batch,
                    ghit_batch,
                    mmiss_batch,
                    preview_ok,
                    preview_tx,
                    last_preview,
                    _lru_limit,
                    t0,
                    abort,
                    placed_total + add,
                    remain_suffix,
                );
                clear_col_unrolled(cols0, x);
            }
        }
        ColGen::Stream(colv) => {
            if profile_enabled {
                let mut enum_time = Duration::ZERO;
                stream_column_candidates_timed(colv, &mut enum_time, |masks| {
                    if abort.load(Ordering::Relaxed) {
                        return;
                    }
                    time_batch.dfs_counts[depth].cand_generated += 1;

                    prof!(profile_enabled, time_batch.dfs_times[depth].assign_cols, {
                        assign_col_unrolled(cols0, x, &masks);
                    });

                    let add = (0..4).map(|c| masks[c].count_ones()).sum::<u32>();
                    let _ = dfs_combine_parallel(
                        depth + 1,
                        cols0,
                        gens,
                        order,
                        threshold,
                        exact_four_only,
                        _memo,
                        local_output_once,
                        global_output_once,
                        _global_memo,
                        map_label_to_color,
                        batch,
                        batch_sender,
                        stat_sender,
                        profile_enabled,
                        time_batch,
                        nodes_batch,
                        leaves_batch,
                        outputs_batch,
                        pruned_batch,
                        lhit_batch,
                        ghit_batch,
                        mmiss_batch,
                        preview_ok,
                        preview_tx,
                        last_preview,
                        _lru_limit,
                        t0,
                        abort,
                        placed_total + add,
                        remain_suffix,
                    );
                    clear_col_unrolled(cols0, x);
                });
                time_batch.dfs_times[depth].gen_candidates += enum_time;
            } else {
                stream_column_candidates(colv, |masks| {
                    if abort.load(Ordering::Relaxed) {
                        return;
                    }
                    assign_col_unrolled(cols0, x, &masks);
                    let add = (0..4).map(|c| masks[c].count_ones()).sum::<u32>();
                    let _ = dfs_combine_parallel(
                        depth + 1,
                        cols0,
                        gens,
                        order,
                        threshold,
                        exact_four_only,
                        _memo,
                        local_output_once,
                        global_output_once,
                        _global_memo,
                        map_label_to_color,
                        batch,
                        batch_sender,
                        stat_sender,
                        profile_enabled,
                        time_batch,
                        nodes_batch,
                        leaves_batch,
                        outputs_batch,
                        pruned_batch,
                        lhit_batch,
                        ghit_batch,
                        mmiss_batch,
                        preview_ok,
                        preview_tx,
                        last_preview,
                        _lru_limit,
                        t0,
                        abort,
                        placed_total + add,
                        remain_suffix,
                    );
                    clear_col_unrolled(cols0, x);
                });
            }
        }
    }

    Ok(())
}

// 検索メイン（並列化＋シングル writer スレッド＋集約 Progress）
#[allow(clippy::too_many_arguments)]
fn run_search(
    base_board: Vec<char>,
    threshold: u32,
    lru_limit: usize,
    outfile: std::path::PathBuf,
    tx: Sender<Message>,
    abort: Arc<AtomicBool>,
    stop_progress_plateau: f32,
    exact_four_only: bool,
    profile_enabled: bool,
) -> Result<()> {
    let info = build_abstract_info(&base_board);
    let colorings = enumerate_colorings_fast(&info);
    if colorings.is_empty() {
        let _ = tx.send(Message::Log(
            "抽象ラベルの4色彩色が存在しないため、探索を終了します。".into(),
        ));
        let _ = tx.send(Message::Finished(Stats::default()));
        return Ok(());
    }
    let _ = tx.send(Message::Log(format!(
        "抽象ラベル={} / 彩色候補={} / 4個消しモード={} / 計測={}",
        info.labels.iter().collect::<String>(),
        colorings.len(),
        if exact_four_only { "ON" } else { "OFF" },
        if profile_enabled { "ON" } else { "OFF" }
    )));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let bmi2 = std::is_x86_feature_detected!("bmi2");
        let popcnt = std::is_x86_feature_detected!("popcnt");
        let _ = tx.send(Message::Log(format!(
            "CPU features: bmi2={} / popcnt={}",
            bmi2, popcnt
        )));
    }

    // 厳密総数（列DP）とメタ構築
    type Meta = (HashMap<char, u8>, [ColGen; W], [u8; W], Vec<usize>);
    let mut metas: Vec<Meta> = Vec::new();
    let mut total = BigUint::zero();
    for assign in &colorings {
        let mut map = HashMap::<char, u8>::new();
        for (i, &lab) in info.labels.iter().enumerate() {
            map.insert(lab, assign[i]);
        }
        let templ = apply_coloring_to_template(&base_board, &map);
        let mut cols: [Vec<TCell>; W] = array_init(|_| Vec::with_capacity(H));
        for x in 0..W {
            for y in 0..H {
                cols[x].push(templ[y * W + x]);
            }
        }
        let mut prod = BigUint::one();
        let mut impossible = false;
        let mut counts: [BigUint; W] = array_init(|_| BigUint::zero());
        let mut max_fill_arr: [u8; W] = [0; W];
        for x in 0..W {
            let cnt = count_column_candidates_dp(&cols[x]);
            if cnt.is_zero() {
                impossible = true;
                break;
            }
            counts[x] = cnt.clone();
            max_fill_arr[x] = compute_max_fill(&cols[x]);
            prod *= cnt;
        }
        if !impossible {
            total += prod;
            let mut order: Vec<usize> = (0..W).collect();
            order.sort_by(|&a, &b| counts[a].cmp(&counts[b]));
            let gens: [ColGen; W] = array_init(|x| build_colgen(&cols[x], &counts[x]));
            metas.push((map, gens, max_fill_arr, order));
        }
    }
    let _ = tx.send(Message::Log(format!(
        "厳密な総組合せ（列制約適用）: {}",
        total
    )));

    // writer
    let (wtx, wrx) = unbounded::<Vec<String>>();
    let tx_for_writer = tx.clone();
    let writer_handle = {
        let outfile = outfile.clone();
        thread::spawn(move || -> Result<()> {
            let mut io_time = Duration::ZERO;
            let file = File::create(&outfile)
                .with_context(|| format!("出力を作成できません: {}", outfile.display()))?;
            let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
            while let Ok(batch) = wrx.recv() {
                let t0 = Instant::now();
                for line in batch {
                    writer.write_all(line.as_bytes())?;
                    writer.write_all(b"\n")?;
                }
                io_time += t0.elapsed();
            }
            writer.flush()?;
            if io_time != Duration::ZERO {
                let mut td = TimeDelta::default();
                td.io_write_total = io_time;
                let _ = tx_for_writer.send(Message::TimeDelta(td));
            }
            Ok(())
        })
    };

    // 集約 Progress
    let (stx, srx) = unbounded::<StatDelta>();
    let t0 = Instant::now();
    let tx_progress = tx.clone();
    let total_clone = total.clone();
    let abort_for_agg = abort.clone();
    let global_output_once: Arc<DU64Set> =
        Arc::new(DU64Set::with_hasher(BuildNoHashHasher::default()));
    // グローバル memo は get を廃止（挿入もしない方針）
    let global_memo: Arc<DU64Map<bool>> =
        Arc::new(DU64Map::with_hasher(BuildNoHashHasher::default()));

    let global_memo_for_agg = global_memo.clone();
    let lru_limit_for_agg = lru_limit;
    let agg_handle = thread::spawn(move || {
        let mut nodes: u64 = 0;
        let mut outputs: u64 = 0;
        let mut done = BigUint::zero();
        let mut pruned: u64 = 0;
        let mut lhit: u64 = 0;
        let mut ghit: u64 = 0;
        let mut mmiss: u64 = 0;
        let mut last_send = Instant::now();

        let mut last_output: u64 = 0;
        let mut progress_at_last_output: f64 = 0.0;
        let plateau: f64 = stop_progress_plateau as f64;

        loop {
            match srx.recv_timeout(Duration::from_millis(500)) {
                Ok(d) => {
                    nodes += d.nodes;
                    outputs += d.outputs;
                    if d.leaves > 0 {
                        done += BigUint::from(d.leaves);
                    }
                    pruned += d.pruned;
                    lhit += d.lhit;
                    ghit += d.ghit;
                    mmiss += d.mmiss;

                    if outputs > last_output {
                        last_output = outputs;
                        if let (Some(td), Some(tt)) = (done.to_f64(), total_clone.to_f64()) {
                            if tt > 0.0 {
                                progress_at_last_output = (td / tt).clamp(0.0, 1.0);
                            }
                        }
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    let dt = t0.elapsed().as_secs_f64();
                    let rate = if dt > 0.0 { nodes as f64 / dt } else { 0.0 };
                    let memo_len = global_memo_for_agg.len();
                    let st = Stats {
                        searching: false,
                        unique: outputs,
                        output: outputs,
                        nodes,
                        pruned,
                        memo_hit_local: lhit,
                        memo_hit_global: ghit,
                        memo_miss: mmiss,
                        total: total_clone.clone(),
                        done: done.clone(),
                        rate,
                        memo_len,
                        lru_limit: lru_limit_for_agg,
                        profile: ProfileTotals::default(),
                    };
                    let _ = tx_progress.send(Message::Finished(st));
                    break;
                }
            }

            if plateau > 0.0 {
                if let (Some(td), Some(tt)) = (done.to_f64(), total_clone.to_f64()) {
                    if tt > 0.0 {
                        let p = (td / tt).clamp(0.0, 1.0);
                        if p - progress_at_last_output >= plateau {
                            let msg = format!(
                                "早期終了: 進捗が {:.1}% 進む間に新規出力がありませんでした（しきい値 {:.1}%）",
                                (p - progress_at_last_output) * 100.0,
                                plateau * 100.0
                            );
                            let _ = tx_progress.send(Message::Log(msg));
                            abort_for_agg.store(true, Ordering::Relaxed);
                        }
                    }
                }
            }

            if last_send.elapsed() >= Duration::from_millis(500) {
                let dt = t0.elapsed().as_secs_f64();
                let rate = if dt > 0.0 { nodes as f64 / dt } else { 0.0 };
                let memo_len = global_memo_for_agg.len();
                let st = Stats {
                    searching: true,
                    unique: outputs,
                    output: outputs,
                    nodes,
                    pruned,
                    memo_hit_local: lhit,
                    memo_hit_global: ghit,
                    memo_miss: mmiss,
                    total: total_clone.clone(),
                    done: done.clone(),
                    rate,
                    memo_len,
                    lru_limit: lru_limit_for_agg,
                    profile: ProfileTotals::default(),
                };
                let _ = tx_progress.send(Message::Progress(st));
                last_send = Instant::now();
            }
        }
    });

    // metas を並列探索
    metas.par_iter().enumerate().try_for_each(
        |(i, (map_label_to_color, gens, max_fill, order))| -> Result<()> {
            if abort.load(Ordering::Relaxed) {
                return Ok(());
            }

            let preview_ok = i == 0;
            let first_x = order[0];
            let mut first_candidates: Vec<[u16; 4]> = Vec::new();
            match &gens[first_x] {
                ColGen::Pre(v) => first_candidates.extend_from_slice(v),
                ColGen::Stream(colv) => {
                    stream_column_candidates(colv, |m| first_candidates.push(m));
                }
            }

            let mut remain_suffix: Vec<u16> = vec![0; W + 1];
            for d in (0..W).rev() {
                remain_suffix[d] = remain_suffix[d + 1] + max_fill[order[d]] as u16;
            }

            let threads = rayon::current_num_threads().max(1);
            if first_candidates.len() >= threads {
                first_candidates
                    .par_iter()
                    .try_for_each(|masks| -> Result<()> {
                        if abort.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                        let mut cols0 = [[0u16; W]; 4];
                        for c in 0..4 {
                            cols0[c][first_x] = masks[c];
                        }
                        let mut memo =
                            ApproxLru::new(lru_limit / rayon::current_num_threads().max(1));
                        let mut local_output_once: U64Set = U64Set::default();
                        let mut batch: Vec<String> = Vec::with_capacity(256);

                        let mut nodes_batch: u64 = 0;
                        let mut leaves_batch: u64 = 0;
                        let mut outputs_batch: u64 = 0;
                        let mut pruned_batch: u64 = 0;
                        let mut lhit_batch: u64 = 0;
                        let mut ghit_batch: u64 = 0;
                        let mut mmiss_batch: u64 = 0;
                        let mut last_preview = Instant::now();

                        let mut time_batch = TimeDelta::default();

                        let placed_first: u32 = (0..4).map(|c| masks[c].count_ones()).sum();
                        let _ = dfs_combine_parallel(
                            1,
                            &mut cols0,
                            gens,
                            order,
                            threshold,
                            exact_four_only,
                            &mut memo,
                            &mut local_output_once,
                            &global_output_once,
                            &global_memo,
                            map_label_to_color,
                            &mut batch,
                            &wtx,
                            &stx,
                            profile_enabled,
                            &mut time_batch,
                            &mut nodes_batch,
                            &mut leaves_batch,
                            &mut outputs_batch,
                            &mut pruned_batch,
                            &mut lhit_batch,
                            &mut ghit_batch,
                            &mut mmiss_batch,
                            preview_ok,
                            &tx,
                            &mut last_preview,
                            lru_limit,
                            t0,
                            &abort,
                            placed_first,
                            &remain_suffix,
                        );

                        if !batch.is_empty() {
                            let _ = wtx.send(batch);
                        }
                        if nodes_batch > 0
                            || leaves_batch > 0
                            || outputs_batch > 0
                            || pruned_batch > 0
                            || lhit_batch > 0
                            || ghit_batch > 0
                            || mmiss_batch > 0
                        {
                            let _ = stx.send(StatDelta {
                                nodes: nodes_batch,
                                leaves: leaves_batch,
                                outputs: outputs_batch,
                                pruned: pruned_batch,
                                lhit: lhit_batch,
                                ghit: ghit_batch,
                                mmiss: mmiss_batch,
                            });
                        }
                        if profile_enabled && time_delta_has_any(&time_batch) {
                            let _ = tx.send(Message::TimeDelta(time_batch));
                        }
                        Ok(())
                    })?;
            } else {
                let second_x = order[1];
                let mut second_candidates: Vec<[u16; 4]> = Vec::new();
                match &gens[second_x] {
                    ColGen::Pre(v) => second_candidates.extend_from_slice(v),
                    ColGen::Stream(colv) => {
                        stream_column_candidates(colv, |m| second_candidates.push(m));
                    }
                }

                second_candidates
                    .par_iter()
                    .try_for_each(|m2| -> Result<()> {
                        if abort.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                        for masks in &first_candidates {
                            if abort.load(Ordering::Relaxed) {
                                break;
                            }

                            let mut cols0 = [[0u16; W]; 4];
                            for c in 0..4 {
                                cols0[c][first_x] = masks[c];
                                cols0[c][second_x] = m2[c];
                            }
                            let mut memo =
                                ApproxLru::new(lru_limit / rayon::current_num_threads().max(1));
                            let mut local_output_once: U64Set = U64Set::default();
                            let mut batch: Vec<String> = Vec::with_capacity(256);

                            let mut nodes_batch: u64 = 0;
                            let mut leaves_batch: u64 = 0;
                            let mut outputs_batch: u64 = 0;
                            let mut pruned_batch: u64 = 0;
                            let mut lhit_batch: u64 = 0;
                            let mut ghit_batch: u64 = 0;
                            let mut mmiss_batch: u64 = 0;
                            let mut last_preview = Instant::now();

                            let mut time_batch = TimeDelta::default();

                            let placed2: u32 = (0..4)
                                .map(|c| masks[c].count_ones() + m2[c].count_ones())
                                .sum();
                            let _ = dfs_combine_parallel(
                                2,
                                &mut cols0,
                                gens,
                                order,
                                threshold,
                                exact_four_only,
                                &mut memo,
                                &mut local_output_once,
                                &global_output_once,
                                &global_memo,
                                map_label_to_color,
                                &mut batch,
                                &wtx,
                                &stx,
                                profile_enabled,
                                &mut time_batch,
                                &mut nodes_batch,
                                &mut leaves_batch,
                                &mut outputs_batch,
                                &mut pruned_batch,
                                &mut lhit_batch,
                                &mut ghit_batch,
                                &mut mmiss_batch,
                                preview_ok,
                                &tx,
                                &mut last_preview,
                                lru_limit,
                                t0,
                                &abort,
                                placed2,
                                &remain_suffix,
                            );

                            if !batch.is_empty() {
                                let _ = wtx.send(batch);
                            }
                            if nodes_batch > 0
                                || leaves_batch > 0
                                || outputs_batch > 0
                                || pruned_batch > 0
                                || lhit_batch > 0
                                || ghit_batch > 0
                                || mmiss_batch > 0
                            {
                                let _ = stx.send(StatDelta {
                                    nodes: nodes_batch,
                                    leaves: leaves_batch,
                                    outputs: outputs_batch,
                                    pruned: pruned_batch,
                                    lhit: lhit_batch,
                                    ghit: ghit_batch,
                                    mmiss: mmiss_batch,
                                });
                            }
                            if profile_enabled && time_delta_has_any(&time_batch) {
                                let _ = tx.send(Message::TimeDelta(time_batch));
                            }
                        }
                        Ok(())
                    })?;
            }
            Ok(())
        },
    )?;

    drop(wtx);
    drop(stx);
    let writer_result = writer_handle
        .join()
        .map_err(|_| anyhow!("writer join error"))?;
    writer_result?;
    agg_handle.join().map_err(|_| anyhow!("agg join error"))?;

    Ok(())
}

// ユーティリティ：配列初期化（const generics）
fn array_init<T, F: FnMut(usize) -> T, const N: usize>(mut f: F) -> [T; N] {
    use std::mem::MaybeUninit;
    let mut data: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    for (i, slot) in data.iter_mut().enumerate() {
        slot.write(f(i));
    }
    unsafe { std::mem::transmute_copy::<[MaybeUninit<T>; N], [T; N]>(&data) }
}

// 列テンプレートから、最大で何マス埋められるか（下から Blank までの連続非 Blank 数）
#[inline(always)]
fn compute_max_fill(col: &[TCell]) -> u8 {
    let mut cnt: u8 = 0;
    for &cell in col.iter().take(H) {
        match cell {
            TCell::Blank => break,
            _ => cnt = cnt.saturating_add(1),
        }
    }
    cnt
}
