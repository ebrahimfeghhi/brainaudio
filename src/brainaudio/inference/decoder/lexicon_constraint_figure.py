"""
Generate a publication-quality PDF figure illustrating the vectorized
lexicon constraint (VectorizedLexiconConstraint).

Run:  python lexicon_constraint_figure.py
Output: lexicon_constraint_figure.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
C_ROOT    = '#E87722'   # orange  – root state
C_NODE    = '#4472C4'   # blue    – interior trie node
C_END     = '#70AD47'   # green   – end state (word complete)
C_SINK    = '#C55A11'   # burnt   – sink state
C_VALID   = '#4472C4'   # blue    – valid transition cell
C_RESET   = '#70AD47'   # green   – reset-to-root transition cell
C_BLANK   = '#FFC000'   # gold    – blank (always allowed)
C_INVAL   = '#F2F2F2'   # gray    – −1 cell
C_STATE   = '#E2EFDA'   # pale green – state tensor cells
C_MASK_T  = '#4472C4'   # blue    – True mask cell
C_MASK_F  = '#F2F2F2'   # gray    – False mask cell
WHITE     = '#FFFFFF'
DARK      = '#222222'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 7.5,
    'pdf.fonttype': 42,   # embed fonts properly
})

# ── example lexicon ──────────────────────────────────────────────────────────
# Words: "hi" → [h(1), i(2), |(4)]    "he" → [h(1), e(3), |(4)]
# Vocab: blank(0), h(1), i(2), e(3), |(4)
# BFS states: ROOT(0), H(1), HI(2), HE(3), HI_END(4), HE_END(5), SINK(6)

INV = -1
table = np.array([
    # ·    h    i    e    |
    [INV,  1,  INV, INV, INV],  # 0: ROOT
    [INV, INV,  2,   3,  INV],  # 1: H
    [INV, INV, INV, INV,  4 ],  # 2: HI
    [INV, INV, INV, INV,  5 ],  # 3: HE
    [INV, INV, INV, INV,  0 ],  # 4: HI_END  ← end state
    [INV, INV, INV, INV,  0 ],  # 5: HE_END  ← end state
    [INV, INV, INV, INV, INV],  # 6: SINK
])
TOKEN_NAMES  = ['·', 'h', 'i', 'e', '|']
STATE_LABELS = ['ROOT', 'H', 'HI', 'HE', 'HI_e', 'HE_e', 'SINK']
END_STATES   = {4, 5}

# State tensor used in the right panel  [B=2, K=2]
state_bk = np.array([[0, 1],
                      [2, 4]])

# ── helpers ───────────────────────────────────────────────────────────────────
def node_circle(ax, cx, cy, label, facecolor, r=0.055, fs=7):
    circ = plt.Circle((cx, cy), r, facecolor=facecolor, edgecolor='white',
                       linewidth=0.8, zorder=4)
    ax.add_patch(circ)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fs,
            color='white', fontweight='bold', zorder=5)


def edge_arrow(ax, x1, y1, x2, y2, label, lc='#555555', ls='-', curved=False, rad=0.0):
    style = f'arc3,rad={rad}' if curved else 'arc3,rad=0'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=lc, lw=1.0,
                                connectionstyle=style), zorder=3)
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    norm = max((dx**2 + dy**2)**0.5, 1e-9)
    ox, oy = -dy / norm * 0.06, dx / norm * 0.06
    if curved:
        ox, oy = ox * 2.5, oy * 2.5
    ax.text(mx + ox, my + oy, label, ha='center', va='center',
            fontsize=6.5, color=lc)


def cell_rect(ax, x, y, w, h, fc, ec='white', lw=0.5):
    r = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.008',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(r)


# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8.0, 3.0), facecolor='white')

ax_trie  = fig.add_axes([0.01, 0.05, 0.26, 0.88])
ax_table = fig.add_axes([0.30, 0.05, 0.30, 0.88])
ax_ops   = fig.add_axes([0.63, 0.05, 0.36, 0.88])

for ax in (ax_trie, ax_table, ax_ops):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# ════════════════════════════════════════════════════════════════════════════
# PANEL A – Trie
# ════════════════════════════════════════════════════════════════════════════
ax_trie.text(0.5, 0.97, '(a) Lexicon Trie', ha='center', va='top',
             fontsize=8, fontweight='bold')

# Node positions
NP = {
    'ROOT':   (0.15, 0.52),
    'H':      (0.50, 0.52),
    'HI':     (0.82, 0.74),
    'HE':     (0.82, 0.30),
    'HI_e':   (0.82, 0.74),   # overlaps HI → drawn as annotation
    'HE_e':   (0.82, 0.30),
}

# Edges
edge_arrow(ax_trie, *NP['ROOT'], *NP['H'],  'h')
edge_arrow(ax_trie, *NP['H'],   *NP['HI'],  'i',  rad= 0.0)
edge_arrow(ax_trie, *NP['H'],   *NP['HE'],  'e',  rad= 0.0)

# HI → HI_end  and  HE → HE_end  shown as "| →" label on node
# (no extra state drawn to keep it compact; end marker on the node itself)

# Nodes
node_circle(ax_trie, *NP['ROOT'], '0\nROOT', C_ROOT, r=0.10, fs=6.5)
node_circle(ax_trie, *NP['H'],    '1\nH',    C_NODE, r=0.08, fs=6.5)

# HI / HE as double rings (end states)
node_circle(ax_trie, *NP['HI'],   '2\nHI',   C_NODE, r=0.08, fs=6.5)
node_circle(ax_trie, *NP['HE'],   '3\nHE',   C_NODE, r=0.08, fs=6.5)

# End state markers to the right of HI and HE
for (nx, ny), word_end, state_id in [
        (NP['HI'], 'hi', 4), (NP['HE'], 'he', 5)]:
    ex, ey = nx, ny   # same position, add "|" edge to the right
    # draw a small end node
    end_x = ex + 0.22
    node_circle(ax_trie, end_x, ey, f'{state_id}\n⊣', C_END, r=0.08, fs=6.5)
    edge_arrow(ax_trie, ex + 0.08, ey, end_x - 0.08, ey, '|', lc='#444')
    ax_trie.annotate('', xy=(0.15, 0.62), xytext=(end_x, ey + 0.08),
                     arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.8,
                                     connectionstyle='arc3,rad=-0.4'), zorder=1)

ax_trie.text(0.73, 0.88, 'auto-reset\nto ROOT', ha='center', fontsize=5.5,
             color='#888', style='italic')

# ════════════════════════════════════════════════════════════════════════════
# PANEL B – Transition Table
# ════════════════════════════════════════════════════════════════════════════
ax_table.text(0.5, 0.97, '(b) Dense Transition Table', ha='center', va='top',
              fontsize=8, fontweight='bold')
ax_table.text(0.5, 0.90, 'transition_table  [S × V]', ha='center', va='top',
              fontsize=6.5, color='#555', style='italic')

n_s, n_v = table.shape
col_label_x0 = 0.32
col_w = 0.11
row_label_x  = 0.05
row_h        = 0.092
row_y0       = 0.83   # top of first data row

# Column headers
for j, tok in enumerate(TOKEN_NAMES):
    cx = col_label_x0 + (j + 0.5) * col_w
    fc = C_BLANK if j == 0 else '#DDDDDD'
    cell_rect(ax_table, col_label_x0 + j * col_w, row_y0 + 0.005, col_w - 0.01,
              0.055, fc=fc)
    ax_table.text(cx, row_y0 + 0.033, tok, ha='center', va='center',
                  fontsize=7, fontweight='bold', color=DARK)

# Rows
for i, slabel in enumerate(STATE_LABELS):
    row_y = row_y0 - (i + 1) * row_h
    cy = row_y + row_h / 2

    # State label badge
    if slabel == 'ROOT':
        badge_c = C_ROOT
    elif i in END_STATES:
        badge_c = C_END
    elif slabel == 'SINK':
        badge_c = C_SINK
    else:
        badge_c = C_NODE
    cell_rect(ax_table, 0.01, row_y + 0.005, 0.27, row_h - 0.01, fc=badge_c)
    ax_table.text(0.145, cy, f'{i}  {slabel}', ha='center', va='center',
                  fontsize=6.5, color='white', fontweight='bold')

    # Table cells
    for j in range(n_v):
        val = table[i, j]
        cx = col_label_x0 + (j + 0.5) * col_w
        cell_x = col_label_x0 + j * col_w

        if j == 0:                          # blank – always valid
            fc = C_BLANK; txt = '✓'; tc = DARK
        elif val == INV:
            fc = C_INVAL; txt = ''; tc = '#aaa'
        elif i in END_STATES and j == 4:    # reset edge |→ROOT
            fc = C_RESET; txt = str(val); tc = WHITE
        else:
            fc = C_VALID; txt = str(val); tc = WHITE

        cell_rect(ax_table, cell_x, row_y + 0.005, col_w - 0.01, row_h - 0.01, fc=fc)
        if txt:
            ax_table.text(cx, cy, txt, ha='center', va='center',
                          fontsize=6.5, color=tc)

# Legend
legend_items = [
    mpatches.Patch(facecolor=C_BLANK, label='blank (always ✓)'),
    mpatches.Patch(facecolor=C_VALID, label='valid → target state'),
    mpatches.Patch(facecolor=C_RESET, label='reset → ROOT'),
    mpatches.Patch(facecolor=C_INVAL, edgecolor='#ccc', label='invalid (−1)'),
]
ax_table.legend(handles=legend_items, loc='lower center',
                fontsize=5.8, frameon=True, ncol=2,
                bbox_to_anchor=(0.5, -0.03))

# ════════════════════════════════════════════════════════════════════════════
# PANEL C – Vectorized batch operations
# ════════════════════════════════════════════════════════════════════════════
ax_ops.text(0.5, 0.97, '(c) Vectorized Batch Constraint', ha='center', va='top',
            fontsize=8, fontweight='bold')

# ── helper: draw a small labeled 2-D tensor grid ─────────────────────────
def draw_tensor(ax, data, x0, y0, cell_w, cell_h, cmap_fn,
                fmt=None, row_labels=None, col_labels=None, title='', fs=6.5):
    R, C = data.shape
    if title:
        ax.text(x0 + C * cell_w / 2, y0 + 0.04, title,
                ha='center', va='bottom', fontsize=fs, fontweight='bold')
    if col_labels:
        for j, cl in enumerate(col_labels):
            ax.text(x0 + (j + 0.5) * cell_w, y0 + 0.025, cl,
                    ha='center', va='bottom', fontsize=fs - 0.5, color='#555')
    for i in range(R):
        for j in range(C):
            val = data[i, j]
            fc = cmap_fn(val, i, j)
            cell_rect(ax, x0 + j * cell_w, y0 - (i + 1) * cell_h,
                      cell_w - 0.012, cell_h - 0.008, fc=fc)
            label = fmt(val) if fmt else str(val)
            tc = WHITE if fc not in (C_INVAL, C_STATE, C_MASK_F, '#F2F2F2') else DARK
            ax.text(x0 + (j + 0.5) * cell_w,
                    y0 - (i + 0.5) * cell_h,
                    label, ha='center', va='center', fontsize=fs, color=tc)
        if row_labels:
            ax.text(x0 - 0.02, y0 - (i + 0.5) * cell_h,
                    row_labels[i], ha='right', va='center',
                    fontsize=fs - 0.5, color='#555')


# ── 1. state[B, K] ──────────────────────────────────────────────────────────
STATE_COLORS = {0: C_ROOT, 1: C_NODE, 2: C_NODE, 3: C_NODE, 4: C_END, 5: C_END, 6: C_SINK}
STATE_NAME_SHORT = {0: 'ROOT', 1: 'H', 2: 'HI', 3: 'HE', 4: 'HI_e', 5: 'HE_e', 6: 'SINK'}

def state_color(val, i, j): return STATE_COLORS.get(int(val), C_NODE)

draw_tensor(ax_ops, state_bk,
            x0=0.02, y0=0.87, cell_w=0.15, cell_h=0.13,
            cmap_fn=state_color,
            fmt=lambda v: STATE_NAME_SHORT.get(int(v), str(v)),
            col_labels=['k=0', 'k=1'],
            row_labels=['b=0', 'b=1'],
            title='state  [B, K]', fs=6.5)

# arrow down
ax_ops.annotate('', xy=(0.175, 0.54), xytext=(0.175, 0.61),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))
ax_ops.text(0.30, 0.575, 'gather rows from\ntransition_table',
            ha='left', va='center', fontsize=6.2, color='#333', style='italic')

# ── 2. gathered rows → mask ──────────────────────────────────────────────────
# For each (b,k) beam, show which tokens are valid (from the table)
#   state[0,0]=ROOT(0): valid tokens = {h(1)}           + blank
#   state[0,1]=H(1):    valid tokens = {i(2), e(3)}     + blank
#   state[1,0]=HI(2):   valid tokens = {|(4)}           + blank
#   state[1,1]=HI_e(4): valid tokens = {|(4) extra}     + blank
#  mask[b, k, v] = True if (transition_table[state[b,k], v] != -1) OR v==blank
mask_data = np.array([
    # ·    h    i    e    |
    [1,   1,   0,   0,   0],   # (b=0,k=0): ROOT
    [1,   0,   1,   1,   0],   # (b=0,k=1): H
    [1,   0,   0,   0,   1],   # (b=1,k=0): HI
    [1,   0,   0,   0,   1],   # (b=1,k=1): HI_e
])

def mask_color(val, i, j):
    if j == 0: return C_BLANK
    return C_MASK_T if val == 1 else C_MASK_F

draw_tensor(ax_ops, mask_data,
            x0=0.02, y0=0.54, cell_w=0.105, cell_h=0.115,
            cmap_fn=mask_color,
            fmt=lambda v: 'T' if v == 1 else '',
            col_labels=TOKEN_NAMES,
            row_labels=['(0,0)', '(0,1)', '(1,0)', '(1,1)'],
            title='mask  [B, K, V]   (≠ −1)', fs=6.5)

# ── 3. update_state arrow + label ───────────────────────────────────────────
ax_ops.plot([0.02, 0.62], [0.11, 0.11], lw=0.6, color='#ccc', ls='--')
ax_ops.text(0.5, 0.96, '', ha='center')   # spacer

# update_state box
box_x, box_y, box_w, box_h = 0.02, 0.01, 0.96, 0.10
box = mpatches.FancyBboxPatch((box_x, box_y), box_w, box_h,
                               boxstyle='round,pad=0.01',
                               facecolor='#EBF3FB', edgecolor='#4472C4', lw=0.8)
ax_ops.add_patch(box)
ax_ops.text(0.5, 0.065,
            "update_state:  state′ = table[state, token]"
            "  ←  single GPU gather for all B·K beams",
            ha='center', va='center', fontsize=6.2, color='#333')

# ── panel dividers ────────────────────────────────────────────────────────────
for xd in [0.285, 0.618]:
    fig.add_artist(plt.Line2D([xd, xd], [0.04, 0.98],
                               transform=fig.transFigure,
                               color='#cccccc', lw=0.8, ls='--'))

# ── compile arrow between panels ─────────────────────────────────────────────
fig.add_artist(plt.Annotation(
    '', xy=(0.305, 0.55), xytext=(0.278, 0.55),
    xycoords='figure fraction', textcoords='figure fraction',
    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2)))
fig.text(0.292, 0.59, 'BFS\ncompile', ha='center', fontsize=6, color='#555',
         style='italic')

out_path = '/home/ebrahim/brainaudio/src/brainaudio/inference/decoder/lexicon_constraint_figure.pdf'
fig.savefig(out_path, bbox_inches='tight', dpi=300)
print(f'Saved → {out_path}')
