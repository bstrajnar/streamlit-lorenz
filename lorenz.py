import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Lorenzov kaos", layout="wide")
st.title("🦋 Lorenzov model – animacija kaosa")

# st.markdown("""
# Interaktivni prikaz Lorenzovega modela v 2D:
# - prikazani sta X in Z os 
# """)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Parametri Lorenzovega sistema")

sigma = st.sidebar.slider("σ (sigma)", 5.0, 20.0, 10.0, 0.5)
rho   = st.sidebar.slider("ρ (rho)", 10.0, 40.0, 28.0, 1.0)
beta  = st.sidebar.slider("β (beta)", 1.0, 5.0, 8/3, 0.1)

dx = st.sidebar.slider(
    "Razlika v začetnem pogoju Δx",
    0.0, 0.1, 0.01, 0.01
)

# ============================================================
# LORENZ MODEL (2D: X vs Z)
# ============================================================
dt = 0.01
N_STEPS = 2000
FRAME_STEP = 5  # koliko korakov na frame

def lorenz_step(x, y, z):
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return x + dxdt*dt, y + dydt*dt, z + dzdt*dt

def integrate(x0):
    xs, zs = [x0[0]], [x0[2]]
    x, y, z = x0
    for _ in range(N_STEPS):
        x, y, z = lorenz_step(x, y, z)
        xs.append(x)
        zs.append(z)
    return np.array(xs), np.array(zs)

x0 = np.array([1.0, 1.0, 1.0])
x1 = np.array([1.00 + dx, 1.0, 1.0])

xs0, zs0 = integrate(x0)
xs1, zs1 = integrate(x1)

# ============================================================
# PLOTLY FRAMES (2D)
# ============================================================
frames = []

for i in range(0, N_STEPS, FRAME_STEP):
    frames.append(
        go.Frame(
            data=[
                # Trajectories
                go.Scatter(
                    x=xs0[:i], y=zs0[:i],
                    mode="lines",
                    line=dict(width=3),
                    name="Začetni pogoj"
                ),
                go.Scatter(
                    x=xs1[:i], y=zs1[:i],
                    mode="lines",
                    line=dict(width=3),
                    name="Začetni pogoj + majhna razlika"
                ),
                # Moving points
                go.Scatter(
                    x=[xs0[i]], y=[zs0[i]],
                    mode="markers",
                    marker=dict(size=18),
                    showlegend=False
                ),
                go.Scatter(
                    x=[xs1[i]], y=[zs1[i]],
                    mode="markers",
                    marker=dict(size=18),
                    showlegend=False
                ),
            ],
            name=str(i)
        )
    )

# ============================================================
# INITIAL FIGURE
# ============================================================
fig = go.Figure(
    data=frames[0].data,
    frames=frames
)

# ============================================================
# LAYOUT WITH PLAY BUTTONS
# ============================================================
fig.update_layout(
    width=900,     # make wider
    height=700,    # make shorter
    xaxis=dict(title="X", range=[-25, 25]),
    yaxis=dict(title="Z", range=[0, 60]),
    title="Projekcija na ravnino X-Z.",
    margin=dict(l=0, r=0, t=50, b=0),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="▶️ Animacija",
                    method="animate",
                    args=[
                        None,
                        dict(frame=dict(duration=40, redraw=True),
                             fromcurrent=True,
                             transition=dict(duration=0))
                    ]
                ),
                dict(
                    label="⏸ Zaustavi",
                    method="animate",
                    args=[
                        [None],
                        dict(frame=dict(duration=0, redraw=False),
                             mode="immediate")
                    ]
                )
            ],
            x=0.05,
            y=1.05
        )
    ]
)

# ============================================================
# SLIDER
# ============================================================
fig.update_layout(
    sliders=[
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[
                        [f.name],
                        dict(frame=dict(duration=0, redraw=True), mode="immediate")
                    ],
                    label=f.name
                )
                for f in frames
            ],
            active=0,
            x=0.1,
            y=0,
            len=0.9
        )
    ]
)

# ============================================================
# DISPLAY
# ============================================================
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# EQUATIONS (DIDACTIC)
# ============================================================
st.subheader("Lorenzove enačbe:")

st.latex(r"""
\begin{aligned}
\frac{dx}{dt} &= \sigma (y - x) \\
\frac{dy}{dt} &= x(\rho - z) - y \\
\frac{dz}{dt} &= xy - \beta z
\end{aligned}
""")

st.markdown("""
👉 **Opomba:**  
Majhna sprememba v začetnem polju sčasoma vodi do popolnoma drugačnega razvoja.
""")
