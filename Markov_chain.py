"""
Matrix-only Markov-chain builder with live Graphviz preview
-----------------------------------------------------------
1. Pick N states in the sidebar.
2. Edit the transition matrix (rows must sum to 1).
3. The right-hand Graphviz diagram redraws after each change.
4. View stationary π and run simulations once the matrix is valid.
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import time

# ─── Graphviz (optional) ─────────────────────────────────────────────────────
try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ModuleNotFoundError:
    GRAPHVIZ_AVAILABLE = False

def colour(f: float) -> str:
    """
    Cool-warm diverging palette
      f = 0.0  → deep blue   (#4575b4)
      f = 0.5  → white       (#f7f7f7)
      f = 1.0  → deep red    (#d73027)
    """
    # clamp just in case
    f = max(0.0, min(1.0, float(f)))

    if f < 0.5:                          # blue → white
        t = f / 0.5                      # 0‥1
        r = int( 69 + t * (247 -  69))   #  69 → 247
        g = int(117 + t * (247 - 117))   # 117 → 247
        b = int(180 + t * (247 - 180))   # 180 → 247
    else:                                # white → red
        t = (f - 0.5) / 0.5              # 0‥1
        r = int(247 + t * (215 - 247))   # 247 → 215
        g = int(247 + t * ( 48 - 247))   # 247 →  48
        b = int(247 + t * ( 39 - 247))   # 247 →  39

    return f"#{r:02x}{g:02x}{b:02x}"

# ─── Page + sidebar ──────────────────────────────────────────────────────────
st.set_page_config("Markov-Chain Builder", "🔗", layout="wide")
st.title("Markov Chain Builder")

# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    n_states = st.number_input("Number of states (N)", 2, 20, 5, 1)

    # 1️⃣  initialise names once
    if "state_names" not in st.session_state or len(st.session_state["state_names"]) != n_states:
        st.session_state["state_names"] = [f"S{i}" for i in range(1, n_states + 1)]

    # 2️⃣  editable table of labels
    with st.expander("Rename states"):
        name_df = pd.DataFrame({"Name": st.session_state["state_names"]})
        edited  = st.data_editor(
            name_df,
            num_rows="fixed",
            column_config={"Name": st.column_config.TextColumn("State name")},
            use_container_width=True,
            key="name_editor",
        )

        # save back (strip blanks, ensure uniqueness)
        new_names = edited["Name"].fillna("").astype(str).str.strip().tolist()
        if len(set(new_names)) == n_states and all(new_names):
            st.session_state["state_names"] = new_names
        else:
            st.warning("Names must be non-empty and unique.")

# -------------------------------------------------------------------------
# Everywhere else just use
states = st.session_state["state_names"]

# ─── Default matrix (equiprobable) ───────────────────────────────────────────
if "matrix_df" not in st.session_state or len(st.session_state["matrix_df"]) != n_states:
    st.session_state["matrix_df"] = pd.DataFrame(
        np.full((n_states, n_states), 1 / n_states), index=states, columns=states
    )

# --- example buttons ---------------------------------------------------------
ex_col1, ex_col2 = st.columns(2)

# 1️⃣ random example (still row-stochastic)
if ex_col1.button("Load random example (N=5)"):
    ex = np.array([
        [0.10, 0.25, 0.25, 0.20, 0.20],
        [0.30, 0.10, 0.20, 0.20, 0.20],
        [0.15, 0.15, 0.30, 0.20, 0.20],
        [0.25, 0.20, 0.20, 0.15, 0.20],
        [0.20, 0.20, 0.25, 0.20, 0.15],
    ])
    st.session_state["matrix_df"] = pd.DataFrame(ex, index=[f"S{i}" for i in range(1,6)],
                                                      columns=[f"S{i}" for i in range(1,6)])
    st.rerun()

# 2️⃣ reversible example (detailed balance)
if ex_col2.button("Load detailed-balance example (N=5)"):
    π = np.array([0.1, 0.15, 0.2, 0.25, 0.3])          # chosen stationary dist
    K = np.array([                                     # symmetric “flux” matrix
        [0.00, 0.04, 0.06, 0.05, 0.05],
        [0.04, 0.00, 0.09, 0.06, 0.06],
        [0.06, 0.09, 0.00, 0.10, 0.08],
        [0.05, 0.06, 0.10, 0.00, 0.14],
        [0.05, 0.06, 0.08, 0.14, 0.00],
    ])

    # ── scale so every row will satisfy  Σ_j P_ij ≤ 1  ──────────────────
    row_ratio = K.sum(1) / π           # how much each row exceeds π_i
    factor    = 0.9 / row_ratio.max()  # 0.9 leaves a safety margin
    K *= factor                        # keep symmetry

    # build reversible transition matrix
    P = (K.T / π).T                    # off-diagonals
    P[np.diag_indices_from(P)] = 1 - P.sum(1)

    st.session_state["matrix_df"] = pd.DataFrame(
        P,
        index=[f"S{i}" for i in range(1, 6)],
        columns=[f"S{i}" for i in range(1, 6)],
    )
    st.rerun()

# ─── Layout: left = editor, right = Graphviz ─────────────────────────────────
col_edit, col_dot = st.columns([2, 1])

with col_edit:
    st.subheader("Transition matrix")
    num_cols = {
        c: st.column_config.NumberColumn(
            label=c, min_value=0.0, max_value=1.0, step=0.01, format="%.3f"
        )
        for c in states
    }
    st.session_state["matrix_df"] = st.data_editor(
        st.session_state["matrix_df"],
        num_rows="fixed",
        column_config=num_cols,
        use_container_width=True,
        key="matrix_editor",
    )

# ─── Graphviz preview (always redraw) ────────────────────────────────────────
with col_dot:
    if GRAPHVIZ_AVAILABLE:
        dot = Digraph(engine="dot")
        for s in states:
            dot.node(s)
        for i, src in enumerate(states):
            for j, tgt in enumerate(states):
                p = st.session_state["matrix_df"].iloc[i, j]
                if p > 0:
                    dot.edge(src, tgt, label=f"{p:.2f}")
        st.graphviz_chart(dot, use_container_width=True)
    else:
        st.info("`graphviz` not installed → `pip install graphviz` for preview.")

# ─── Validate & compute π ────────────────────────────────────────────────────
M = st.session_state["matrix_df"].to_numpy(float)
if np.allclose(M.sum(1), 1, atol=1e-6):
    w, v = np.linalg.eig(M.T)
    vec = np.abs(np.real(v[:, np.argmin(np.abs(w - 1))]))
    pi = vec / vec.sum()
    st.session_state["stationary"] = pi
    
    #st.header("Stationary distribution π")
    #st.bar_chart(pd.Series(pi, index=states))

    with st.expander("Simulate random walk"):
        start = st.selectbox("Start state", states)
        steps = st.number_input("Steps", 100, 50000, 2000, 100, key="sim_steps")

        interval   = 100                           # take a snapshot every 100 steps
        frames     = steps // interval + 1         # +1 for the final step (may duplicate)
        emp_tot    = np.zeros((frames, n_states))  # right shape
        traj_tot   = []                            # store only the frame states
        frame_idx  = 0                             # row cursor
        
        if st.button("Run live simulation"):
            M = st.session_state["matrix_df"].to_numpy(float)
            cur = states.index(start)
            counts = np.zeros(n_states, dtype=int)
            emp_tot  = np.zeros((steps + 1, n_states)) 
            
            traj = [cur] 
            traj_tot = [cur]

            prog = st.progress(0, text="Running…")
            dot_ph = st.empty()              
            
            for step in range(1, steps+1):
                counts[cur] += 1
                cur = np.random.choice(n_states, p=M[cur])
                
                if step % interval == 0 or step == steps:
                    traj_tot.append(cur)  
                    empirical = counts / step
                    emp_tot[frame_idx] = empirical
                    frac = counts / step
                    frame_idx += 1 
                    if GRAPHVIZ_AVAILABLE:
                        dot = Digraph(engine="dot")
                        dot.attr("graph", splines="true")
                        dot.attr("node",
                                  shape="ellipse", fixedsize="true",
                                  width="1.0", height="0.6",
                                  fontsize="10", penwidth="1.5")
                        dot.attr("edge", arrowsize="0.6", penwidth="1.2")
                        
                        for i, s in enumerate(states):
                            dot.node(
                                s,
                                style="filled",
                                fillcolor=colour(frac[i]),
                                fontcolor="#000",
                            )
                        for i, src in enumerate(states):
                            for j, tgt in enumerate(states):
                                p = M[i, j]
                                if p > 0:
                                    dot.edge(src, tgt, label=f"{p:.2f}")
                        dot_ph.graphviz_chart(dot, use_container_width=False)
                    prog.progress(step/steps, text=f"{step}/{steps} steps")
            prog.empty()  # clear progress bar
                
            st.dataframe(
                pd.DataFrame(
                    {"Empirical": empirical.round(3),
                 "π": st.session_state.get("stationary", pi).round(3)
                    },
                    index=states,
                ),
                use_container_width=True,
            )

            #traj_df = pd.DataFrame({"Step": range(len(traj)), "State": traj})
            #small_chart = (
            #    alt.Chart(traj_df)
            #    .mark_line(interpolate="step-after", strokeWidth=1)   # thinner line
            #    .encode(
            #        x="Step:Q",
            #        y=alt.Y("State:Q", axis=alt.Axis(tickMinStep=1, title=None)),
            #    )
            #    .properties(height=50)         # ← 150 px tall; adjust to taste
            #)

            #st.altair_chart(small_chart, use_container_width=False)
            #print(traj)
            
            st.session_state["traj"]      = traj_tot          # 🔒 keep the full sequence
            st.session_state["frame_idx"] = 0             # 1-step playback cursor 
            st.session_state["emp_series"]  = emp_tot
else:
    st.warning("Each row must sum to 1 to compute π and run simulations.")
    
# ─────────────────────────────────────────────────────────────────────────
# Trajectory playback – with persistent placeholders
# ─────────────────────────────────────────────────────────────────────────
if "traj" in st.session_state:

    st.session_state.setdefault("playing",   False)
    st.session_state.setdefault("frame_idx", 0)

    # Start / Stop buttons
    cols = st.columns(2)
    if cols[0].button("▶ Start"):
        st.session_state["playing"] = True
    if cols[1].button("⏸ Stop"):
        st.session_state["playing"] = False

    k = st.session_state["frame_idx"]
    cur_state_idx = st.session_state["traj"][k]

    st.caption(
        f"(accelerate 100 times)"
        f"Frame {k} / {len(st.session_state['traj'])-1} – "
        f"state **{states[cur_state_idx]}**"
    )

    # ➊ create placeholders once
    graph_ph = st.empty()
    bar_ph    = st.empty()

    # ➋ always redraw both visuals
    if GRAPHVIZ_AVAILABLE:
        dot = Digraph(engine="dot")
        dot.attr("graph", splines="true")
        dot.attr("node", shape="ellipse", fixedsize="true",
                 width="1.0", height="0.6", fontsize="10", penwidth="1.5")
        dot.attr("edge", arrowsize="0.6", penwidth="1.2")

        # highlight current node
        for i, s in enumerate(states):
            fill = "#ffd700" if i == cur_state_idx else "#e0eaff"
            dot.node(s, style="filled", fillcolor=fill)

        M_play = st.session_state["matrix_df"].to_numpy(float)
        for i, src in enumerate(states):
            for j, tgt in enumerate(states):
                p = M_play[i, j]
                if p > 0:
                    dot.edge(src, tgt, label=f"{p:.2f}")

        graph_ph.graphviz_chart(dot, use_container_width=False)

    # bar chart
    if (
        "emp_series" in st.session_state
        and "stationary" in st.session_state
        and k < len(st.session_state["emp_series"])
    ):
        emp_k = st.session_state["emp_series"][k-1]
        bar_df = pd.DataFrame(
            {"State":      states,
             "Stationary π": st.session_state["stationary"],
             "Empirical π":  emp_k,}
        )
        long = bar_df.melt(id_vars="State", var_name="Type", value_name="Prob")
        
        bar_chart = (
            alt.Chart(long)
            .mark_bar(width=20)
            .encode(
            x=alt.X("State:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset="Type:N",
            y=alt.Y("Prob:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
            color=alt.Color("Type:N", title=None), 
            )
            .properties(height=150)
        )
        
        bar_ph.altair_chart(bar_chart, use_container_width=True)

    # ➌ auto-advance after drawing
    if st.session_state["playing"]:
        time.sleep(2)
        st.session_state["frame_idx"] = (k + 1) % len(st.session_state["traj"])
        st.rerun()