import streamlit as st
import numpy as np
import plotly.graph_objects as go
from tpd_locations_nd import ep_location, tpd_location
from pt_peaks_MODEL import peak_location

# Create a title for the App
st.title("EP and TPD Locations")

# Link to preprint of the paper
st.markdown("[Unification of Exceptional Points and Transmission Peak Degeneracies in a Highly Tunable Magnon-Photon Dimer](https://arxiv.org/abs/2506.09141)")

# Create sidebar with the Parameters
st.sidebar.header("System Parameters")
kappa_tilde_c = st.sidebar.slider(r"$\tilde{\kappa}_c$", 0.0, 2.5, 0.68, step=0.01)

phi_labels = {
    "0": 0,
    "π/8": np.pi / 8,
    "π/4": np.pi / 4,
    "3π/8": 3 * np.pi / 8,
    "π/2": np.pi / 2,
    "5π/8": 5 * np.pi / 8,
    "3π/4": 3 * np.pi / 4,
    "7π/8": 7 * np.pi / 8,
    "π": np.pi,
    "9π/8": 9 * np.pi / 8,
    "5π/4": 5 * np.pi / 4,
    "11π/8": 11 * np.pi / 8,
    "3π/2": 3 * np.pi / 2,
    "13π/8": 13 * np.pi / 8,
    "7π/4": 7 * np.pi / 4,
    "15π/8": 15 * np.pi / 8,
    "2π": 0
}

phi_label = st.sidebar.select_slider("Φ - Coupling Phase", options=list(phi_labels.keys()), value="0")
phi = phi_labels[phi_label]
# phi = st.sidebar.slider("Φ - Coupling Phase (rad)", 0.0, (2 * np.pi), 0.0, step=0.01) # Option for if we want a continuous slider

# Define x and y axis for the plot (x is Delta_tilde_kappa and y is Delta_tilde_f)
x1 = np.linspace(-4, 4, 500)
y1 = np.linspace(-4, 4, 500)
X, Y = np.meshgrid(x1, y1)

# Define modulus squared of Delta_tilde_lambda in terms of X and Y
Delta_tilde_lambda = np.sqrt(-Y ** 2 + 2j * Y * X + X ** 2 - 4 * np.exp(1j * phi))
Delta_tilde_lambda_mod_squared = abs(Delta_tilde_lambda) ** 2

# Petermann Factor
K_2_tilde = (Y ** 2 + X ** 2 + Delta_tilde_lambda_mod_squared + 4) / (2 * Delta_tilde_lambda_mod_squared)

# Contour values
tilde_q = ((kappa_tilde_c - X) * (Delta_tilde_lambda ** 2).imag) / 8
tilde_p = (((kappa_tilde_c - X) ** 2) + (Delta_tilde_lambda ** 2).real) / 4
disc = -4 * (tilde_p ** 3) - 27 * (tilde_q ** 2)
instability = (Delta_tilde_lambda.real) - (kappa_tilde_c - X)
if phi == 0 or phi == 2 * np.pi:
    min_petermann = X
elif phi == np.pi:
        min_petermann = Y
else:
        min_petermann = (-1 / np.tan(phi / 2)) * X - Y

# Compute degeneracy locations
eps = ep_location(phi)
tpds = tpd_location(phi, kappa_tilde_c)

# Dictionary mapping degeneracy type to color and marker
color_marker_dict = {
    "PRIMARY_EP": ("red", "x"),
    "PRIMARY_TPD": ("red", "circle"),
    "SECONDARY_EP": ("gray", "x"),
    "SECONDARY_TPD": ("gray", "circle"),
    "ROGUE_TPD": ("gray", "diamond")
}

# Plot Petermann Factor Color Map
fig1 = go.Figure()
fig1.add_trace(go.Heatmap(
    x = x1,
    y = y1,
    z = K_2_tilde,
    colorscale='plasma',
    zmin = 1.0, 
    zmax = 1.6,
    colorbar = dict(title="Petermann Factor")
))

# Function for plotting contours
def plot_contours(contour, color, linestyle, linewidth, label):
    fig1.add_trace(go.Contour(
    z = contour,
    x = x1,
    y = y1,
    showscale = False,
    contours = dict(start=0, end=0, size=1, coloring='none'),
    line = dict(color=color, dash=linestyle, width=linewidth),
    name = label
))

# Plot the contours
plot_contours(tilde_q, 'magenta', 'dash', 3, '𝑞̃ = 0')
plot_contours(disc, 'cyan', 'dash', 3, 'Disc = 0')
plot_contours(instability, 'chartreuse', 'dash', 3, 'Instability')
plot_contours(min_petermann, 'white', 'dash', 3, 'Petermann Factor = 1')

# Function to plot the degeneracies
def plot_degeneracies(degeneracies):
    # Track which types have already been plotted to avoid duplicates in the legend
    plotted_types = set()

    # Plot each degeneracy with proper labels, colors, and markers
    for degen in degeneracies:
        dtype = degen.degeneracy_type
        
        if dtype not in plotted_types:
            label_list = dtype.name.split("_")
            label_list[0] = label_list[0].title()
            label = " ".join(label_list)
        else:
            label = None

        color, marker = color_marker_dict.get(dtype.name)
        fig1.add_trace(go.Scatter(
            x = [degen.Delta_tilde_kappa],
            y = [degen.Delta_tilde_f],
            mode = 'markers',
            marker = dict(color=color, symbol=marker, size=15),
            name = label,
            showlegend = (label != None)
        ))

        plotted_types.add(dtype)

# Add EPs and TPDs to the plot
plot_degeneracies(eps)
plot_degeneracies(tpds)

fig1.update_layout(
    xaxis_title= '𝛥̃ₖ',
    yaxis_title= '𝛥̃𝑓',
    xaxis_title_font=dict(size=30),
    yaxis_title_font=dict(size=30),
    legend=dict(
        bgcolor = 'lightgrey',
        orientation='h',
        yanchor='top',
        y = -0.20,
        xanchor='center',
        x = 0.5,
        font=dict(size=20)
        ),
    margin=dict(t=20),
    width=650,
    height=650,
)

st.plotly_chart(fig1, use_container_width=True)

# Plot peak splitting of primary TPD
# Set J and f_c to 1 and 0 respectively
J = 1.0
f_c = 0.0

# Find Location of Primary TPD
primary_tpd = None
for degen in tpds:
    if degen.degeneracy_type.name == "PRIMARY_TPD":
        primary_tpd = degen
        break
if primary_tpd is not None:
    dk_center = primary_tpd.Delta_tilde_kappa
    df_center = primary_tpd.Delta_tilde_f
else:
    dk_center = 0
    df_center = 0

# Create different x axes depending on phi
if phi == 0:
    x2_label = '𝛥̃ₖ'
    x2 = np.linspace(dk_center - 1, dk_center + 1, 2000)
    delta_f = 0
elif phi == np.pi:
    x2_label = '𝛥̃𝑓'
    x2 = np.linspace(df_center - 1, df_center + 1, 2000)
    delta_kappa = 0
# elif abs(phi - np.pi) <= np.pi / 8:
#     x2_label = '𝛥̃𝑓'
#     if phi > np.pi:
#         x2 = np.linspace(df_center - 0.15, df_center + 0.25, 2000)
#     else:
#         x2 = np.linspace(df_center + 0.25, df_center - 0.15, 2000)
else:
    x2_label = '𝛥̃ₖ'
    x2 = np.linspace(dk_center - 0.15, dk_center + 0.25, 2000)

# Set up nu+, nu- and nu_0
nu_plus = np.full_like(x2, np.nan)
nu_minus = np.full_like(x2, np.nan)
nu_0 = np.full_like(x2, np.nan)

# Calculate peak_locations for each x value
for i, x in enumerate(x2):
    if phi == 0:
        result = peak_location(J, f_c, kappa_tilde_c, delta_f, x, phi)
    elif phi == np.pi:
        result = peak_location(J, f_c, kappa_tilde_c, x, delta_kappa, phi)
    # elif abs(phi - np.pi) <= np.pi / 8:
    #     result = peak_location(J, f_c, kappa_tilde_c, x, (2 * np.sin(phi)) / x, phi)
    else:
        result = peak_location(J, f_c, kappa_tilde_c, (2 * np.sin(phi)) / x, x, phi)

    if len(result) == 2:
        nu_plus[i] = result[0]
        nu_minus[i] = result[1]
    else:
        nu_0[i] = result[0]


fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=x2, y=nu_plus, mode='lines', name='ν±', legendgroup='nu_pm', line=dict(width=4, color='black')))
fig2.add_trace(go.Scatter(x=x2, y=nu_minus, mode='lines', showlegend=False, line=dict(width=4, color='black')))
fig2.add_trace(go.Scatter(x=x2, y=nu_0, mode='lines', showlegend=False, line=dict(width=4, color='black')))

fig2.update_layout(
    title=f"Primary TPD Peak Splitting",
    title_font = dict(size=25),
    xaxis_title = x2_label,
    yaxis_title = "Frequency [arb.]",
    xaxis_title_font = dict(size=30),
    yaxis_title_font = dict(size=30),
    height=500,
    legend=dict(font=dict(size=25)),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)

st.plotly_chart(fig2, use_container_width=True)