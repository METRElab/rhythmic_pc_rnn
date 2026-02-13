import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib import colors
import torch

# Set the visual style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12


def generate_input_visualization(duration=4.0, tempo=0.5, dt=0.01, savepath="input_visualization.png"):
    """
    Generate a visualization of the input signals used in the model:
    - Triangular vestibular input
    - Discrete auditory events

    This corresponds to the training phase stimuli.
    """
    # Generate input sequences
    n_steps = int(duration / dt)
    t = np.arange(0, duration, dt)

    # Beat sequence
    beat_times = np.arange(0, duration, tempo)
    beat_indices = (beat_times / dt).astype(int)
    beat_sequence = np.zeros(n_steps)
    beat_sequence[beat_indices] = 1

    # Create a triangular wave for vestibular input
    frequency = 1 / tempo
    sawtooth = 2 * (t * frequency - np.floor(0.5 + t * frequency))
    vestibular_tri = 1 - 2 * np.abs(sawtooth)

    # Create figure
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot auditory events
    ax[0].stem(t, beat_sequence, linefmt='r-', markerfmt='ro', basefmt='r-',
               label='Auditory Events')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_title('Auditory Input', fontweight='bold')
    ax[0].set_ylim(-0.1, 1.1)

    # Plot vestibular input
    ax[1].plot(t, vestibular_tri, 'b-', label='Vestibular Input', linewidth=2)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_title('Vestibular Input', fontweight='bold')

    # Add vertical lines at beat times
    for beat_time in beat_times:
        if beat_time < duration:
            ax[0].axvline(x=beat_time, color='r', linestyle='--', alpha=0.3)
            ax[1].axvline(x=beat_time, color='r', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Input visualization saved to: {savepath}")

    # Create caption
    caption = """
    Figure X: Synchronized multimodal input signals used during network training. 
    (A) Discrete auditory events occurring at regular intervals (tempo = {tempo_val} s). 
    (B) Continuous triangular vestibular signal simulating the sensation of rhythmic movement. 
    The vestibular signal completes one cycle between each auditory event, modeling the 
    sensorimotor experience during rhythmic activities such as walking or bouncing. Vertical 
    dashed lines indicate the temporal alignment between auditory events and the peaks of 
    the vestibular cycle.
    """.format(tempo_val=tempo)

    return caption

def generate_testing_conditions_visualization(duration=4.0, tempo=0.5, dt=0.01, savepath="testing_visualization.png"):
    """
    Visualize the two testing conditions:
    1. Auditory-only (vestibular precision set to zero)
    2. Complete input removal (both precisions set to zero)
    """
    # Generate input sequences
    n_steps = int(duration / dt)
    t = np.arange(0, duration, dt)

    # Beat sequence
    beat_times = np.arange(0, duration, tempo)
    beat_indices = (beat_times / dt).astype(int)
    beat_sequence = np.zeros(n_steps)
    beat_sequence[beat_indices] = 1

    # Create a triangular wave for vestibular input
    frequency = 1 / tempo
    sawtooth = 2 * (t * frequency - np.floor(0.5 + t * frequency))
    vestibular_tri = 1 - 2 * np.abs(sawtooth)

    # Create hypothetical model output under two conditions
    # Note: These are simulated outputs, not actual model outputs
    # Auditory-only condition: Model still predicts vestibular signal
    vest_pred_auditory_only = vestibular_tri * 0.85 + np.random.normal(0, 0.05, n_steps)

    # No-input condition: Internal rhythm generation with decay
    decay_factor = np.exp(-np.linspace(0, 2, n_steps))
    vest_pred_no_input = vestibular_tri * decay_factor + np.random.normal(0, 0.1, n_steps)

    # Create figure with 3 rows
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])

    # Reference condition (training) - left side
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    # Auditory-only condition - right top
    ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax1)

    # No-input condition - bottom row
    ax5 = fig.add_subplot(gs[2, :], sharex=ax1)

    # Reference condition (training)
    ax1.stem(t, beat_sequence, linefmt='r-', markerfmt='ro', basefmt='r-',
             label='Auditory Events')
    ax1.set_title('Training Phase: Full Input', fontweight='bold')
    ax1.set_ylabel('Auditory\nAmplitude')
    ax1.set_ylim(-0.1, 1.1)

    ax2.plot(t, vestibular_tri, 'b-', label='Vestibular Input', linewidth=2)
    ax2.set_ylabel('Vestibular\nAmplitude')

    # Auditory-only condition (πV = 0)
    ax3.stem(t, beat_sequence, linefmt='r-', markerfmt='ro', basefmt='r-')
    ax3.set_title('Testing Phase 1: Auditory-Only (πV = 0)', fontweight='bold')
    ax3.set_ylim(-0.1, 1.1)

    ax4.plot(t, np.zeros_like(t), 'b--', alpha=0.3, label='No vestibular input')
    ax4.plot(t, vest_pred_auditory_only, 'g-', label='Predicted vestibular', linewidth=2)
    ax4.legend(loc='upper right', frameon=True)

    # No-input condition (πA = πV = 0)
    ax5.plot(t, np.zeros_like(t), 'r--', alpha=0.3, label='No auditory input')
    ax5.plot(t, np.zeros_like(t), 'b--', alpha=0.3, label='No vestibular input')
    ax5.plot(t, vest_pred_no_input, 'g-', label='Internal rhythm generation', linewidth=2)
    ax5.set_title('Testing Phase 2: No Input (πA = πV = 0)', fontweight='bold')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Generated\nAmplitude')
    ax5.legend(loc='upper right', frameon=True)

    # Add vertical lines at beat times
    for beat_time in beat_times:
        if beat_time < duration:
            ax1.axvline(x=beat_time, color='r', linestyle='--', alpha=0.3)
            ax2.axvline(x=beat_time, color='r', linestyle='--', alpha=0.3)
            ax3.axvline(x=beat_time, color='r', linestyle='--', alpha=0.3)
            ax4.axvline(x=beat_time, color='r', linestyle='--', alpha=0.3)
            ax5.axvline(x=beat_time, color='r', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Testing conditions visualization saved to: {savepath}")

    # Create caption
    caption = """
    Figure X: Experimental testing conditions for evaluating rhythm perception and generation.
    (Top Left) Training phase with synchronized auditory events and vestibular signals.
    (Top Right) Testing condition 1: Auditory-only input where vestibular precision (πV) is set to zero,
    causing the network to generate vestibular predictions (green line) based solely on auditory cues.
    (Bottom) Testing condition 2: Complete input removal where both auditory and vestibular precisions 
    (πA and πV) are set to zero, testing the network's ability to maintain internal rhythm generation
    through learned dynamics alone. The decay in amplitude represents the expected gradual loss of 
    temporal precision without external sensory reinforcement.
    """

    return caption

def draw_node(ax, x, y, radius, label, facecolor='white', edgecolor='black', textcolor='black'):
    """Helper function to draw a node with text"""
    circle = plt.Circle((x, y), radius, fill=True, facecolor=facecolor, edgecolor=edgecolor, zorder=2)
    ax.add_patch(circle)
    text = ax.text(x, y, label, ha='center', va='center', color=textcolor, fontweight='bold', zorder=3)
    text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    return circle

def draw_arrow(ax, start, end, color='black', width=0.01, style='->', label=None, label_pos=0.5):
    """Helper function to draw an arrow with optional label"""
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, color=color,
        connectionstyle='arc3,rad=0.1', linewidth=1.5, shrinkA=5, shrinkB=5, zorder=1
    )
    ax.add_patch(arrow)

    if label:
        # Calculate position for label
        mid_x = start[0] + label_pos * (end[0] - start[0])
        mid_y = start[1] + label_pos * (end[1] - start[1])
        text = ax.text(mid_x, mid_y, label, ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                       zorder=3)
        text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    return arrow

def generate_precision_modulation_figure(savepath="precision_modulation.png"):
    """
    Create a visualization showing how precision parameters modulate prediction errors
    and influence network updates.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Create a background rectangle for the sensory pathway
    rect = Rectangle((1, 1), 8, 8, linewidth=2, edgecolor='gray',
                     facecolor='lightgray', alpha=0.2, zorder=0)
    ax.add_patch(rect)

    # Draw nodes
    draw_node(ax, 3, 7, 0.6, "x", facecolor='lightskyblue')
    draw_node(ax, 7, 7, 0.6, "μ", facecolor='lightgreen')
    draw_node(ax, 5, 5, 0.6, "ε", facecolor='salmon')
    draw_node(ax, 3, 3, 0.6, "A", facecolor='gold') # Actual input

    # Draw connections
    draw_arrow(ax, (3.6, 7), (6.4, 7), color='blue', label="Prediction")
    draw_arrow(ax, (7, 6.4), (5.5, 5.5), color='green', style='->')
    draw_arrow(ax, (3, 3.6), (4.5, 4.5), color='orange', label="Input")

    # Precision modulation
    draw_arrow(ax, (5, 5.6), (3.5, 6.8), color='red', style='<-', label="π × ε", label_pos=0.3)

    # Add "π = 0" indicator
    pi_text = ax.text(5, 2, "Precision: π = 1\n(normal input processing)",
                      ha='center', va='center', fontsize=14, fontweight='bold',
                      bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

    # Add "π = 0" indicator
    pi_zero_text = ax.text(8, 4, "π = 0\n(input effectively removed)",
                           ha='center', va='center', fontsize=12,
                           bbox=dict(facecolor='lightgray', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.3'))

    # Add a red X through the arrows for π = 0
    ax.plot([4.2, 5.8], [4.2, 5.8], 'r-', lw=3, alpha=0.7, zorder=3)
    ax.plot([5.8, 4.2], [4.2, 5.8], 'r-', lw=3, alpha=0.7, zorder=3)

    # Add title
    ax.set_title("Precision-based Input Modulation", fontsize=16, fontweight='bold', pad=20)

    # Add explanation text
    explanation = """
    When precision (π) = 0:
    • Prediction errors from this input are ignored
    • No state updates occur based on this input
    • Network must rely on internal dynamics or other inputs
    """
    ax.text(5, 9, explanation, ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Precision modulation figure saved to: {savepath}")

    # Create caption
    caption = """
    Figure X: Precision-based sensory input modulation in the predictive coding framework.
    The diagram illustrates how precision parameters (π) effectively control the influence of 
    sensory inputs on the network's state updates. When precision is set to normal values (π = 1), 
    prediction errors drive updates to the network state. When precision is set to zero (π = 0), 
    the corresponding sensory prediction errors no longer influence network dynamics, effectively 
    removing that input stream. This mechanism allows systematic manipulation of vestibular and 
    auditory inputs during different testing phases without altering the network's architecture.
    """

    return caption

def generate_network_dynamics_figure(savepath="network_dynamics.png"):
    """
    Create a figure illustrating the core network dynamics:
    - Fast state optimization process (error minimization)
    - Slow weight learning process (Hebbian-like learning)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Panel 1: Fast state optimization
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    # Create background
    rect = Rectangle((1, 1), 8, 8, linewidth=2, edgecolor='gray',
                     facecolor='lightblue', alpha=0.1, zorder=0)
    ax1.add_patch(rect)

    # Draw nodes
    x_node = draw_node(ax1, 3, 5, 0.6, "x", facecolor='lightskyblue')
    mu_node = draw_node(ax1, 7, 5, 0.6, "μ", facecolor='lightgreen')
    eps_node = draw_node(ax1, 5, 7, 0.6, "ε", facecolor='salmon')

    # Draw connections for state update
    draw_arrow(ax1, (3.6, 5), (6.4, 5), color='blue', label="Prediction")
    draw_arrow(ax1, (7, 5.6), (5.5, 6.8), color='green')
    draw_arrow(ax1, (5, 6.4), (3.5, 5.4), color='red', label="State\nUpdate")

    # Add equation for fast dynamics
    fast_eq = r"$\dot{x}_i = -\varepsilon_i + \varepsilon_V \theta_{Vi} f'(x_i) + \varepsilon_A \theta_{Ai} f'(x_i)$"
    ax1.text(5, 2, fast_eq, ha='center', va='center', fontsize=14,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

    ax1.set_title("Fast State Optimization\n(Within Timestep)", fontsize=14, fontweight='bold', pad=15)

    # Panel 2: Slow weight learning
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Create background
    rect = Rectangle((1, 1), 8, 8, linewidth=2, edgecolor='gray',
                     facecolor='lightgreen', alpha=0.1, zorder=0)
    ax2.add_patch(rect)

    # Draw nodes (t-1 and t)
    x1_node = draw_node(ax2, 2, 7, 0.5, "xⱼᵗ⁻¹", facecolor='lightskyblue')
    x2_node = draw_node(ax2, 2, 3, 0.5, "xᵢᵗ", facecolor='lightskyblue')
    eps_node = draw_node(ax2, 5, 3, 0.5, "εᵢᵗ", facecolor='salmon')

    # Draw weight connection
    weight_arrow = FancyArrowPatch(
        (2.2, 6.5), (2.2, 3.5), arrowstyle='->', color='purple',
        connectionstyle='arc3,rad=-0.1', linewidth=2, shrinkA=10, shrinkB=10, zorder=1
    )
    ax2.add_patch(weight_arrow)

    # Add weight label
    ax2.text(1.7, 5, "θⱼᵢ", ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Draw error feedback to weight
    draw_arrow(ax2, (4.5, 3), (2.5, 5), color='red', style='<-', label="Update\nWeight")

    # Add equation for weight update
    weight_eq = r"$\Delta\theta_{ji} \propto \varepsilon_i f(x_j)$"
    ax2.text(6, 6, weight_eq, ha='center', va='center', fontsize=14,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

    ax2.set_title("Slow Weight Learning\n(Across Timesteps)", fontsize=14, fontweight='bold', pad=15)

    # Add overall title
    fig.suptitle("Dual Timescale Learning in the Predictive Coding Network",
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Network dynamics figure saved to: {savepath}")

    # Create caption
    caption = """
    Figure X: Dual timescale learning processes in the predictive coding network.
    (A) Fast state optimization: Within each timestep, hidden state values (x) are rapidly 
    updated to minimize current prediction errors (ε) through gradient descent. This process 
    implements Bayesian inference by finding the most likely internal state given sensory evidence.
    (B) Slow weight learning: Across multiple timesteps, connection weights (θ) are gradually 
    adjusted according to a Hebbian-like learning rule where weight changes are proportional to 
    the product of prediction errors and presynaptic activity. This implements long-term learning
    of temporal patterns in the input, enabling the network to predict future events based on past
    experience.
    """

    return caption


if __name__ == "__main__":
    # Generate all figures
    input_caption = generate_input_visualization(savepath="figure1_input_visualization.png")
    print("\nCaption for Figure 1:")
    print(input_caption)

    testing_caption = generate_testing_conditions_visualization(savepath="figure2_testing_conditions.png")
    print("\nCaption for Figure 2:")
    print(testing_caption)

    precision_caption = generate_precision_modulation_figure(savepath="figure3_precision_modulation.png")
    print("\nCaption for Figure 3:")
    print(precision_caption)

    dynamics_caption = generate_network_dynamics_figure(savepath="figure4_network_dynamics.png")
    print("\nCaption for Figure 4:")
    print(dynamics_caption)