import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Arc
import io
from PIL import Image
from IPython.display import Image as animdisp

def get_peaks_LCR(i_0, R, X_L, X_C):
    v_R0 = i_0 * R
    v_L0 = i_0 * X_L
    v_C0 = i_0 * X_C
    v_0 = np.sqrt(v_R0**2 + (v_C0 - v_L0)**2)
    return v_0, v_R0, v_L0, v_C0

def get_ac_case(X_L, X_C):
    if X_L == X_C:
        return 'LCR-resonance'
    elif X_L > X_C:
        return 'LCR-inductive'
    else:
        return 'LCR-capacitive'
    
def calculate_ac(theta, phi, i_0, v_0, v_R0, v_L0, v_C0):
    i = i_0 * np.sin(theta + phi)
    v = v_0 * np.sin(theta)
    v_R = v_R0 * np.sin(theta + phi)
    v_L = v_L0 * np.sin(theta + phi + np.pi/2)
    v_C = v_C0 * np.sin(theta + phi - np.pi/2)
    return i, v, v_R, v_L, v_C

def color_scheme():
    return {'I':   '#da1e37',
            'V':   '#0d47a1',
            'V_R': '#7b2cbf',
            'V_C': '#ff4d6d',
            'V_L': '#008000'}

def setup_canvas():
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]}, figsize=(8.5,3), dpi=150)
    fig.tight_layout()
    for pos in ['right', 'top', 'bottom', 'left']:
        ax[0].spines[pos].set_visible(False)
        ax[1].spines[pos].set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    return fig, ax

def wavesamples(theta_1, phi, i_0, v_0, v_R0, v_L0, v_C0):
    wavesample_i = (theta_1, i_0 * np.sin(theta_1 + phi))
    wavesample_v = (theta_1, v_0 * np.sin(theta_1))
    wavesample_vR = (theta_1, v_R0 * np.sin(theta_1 + phi))
    wavesample_vL = (theta_1, v_L0 * np.sin(theta_1 + phi + np.pi/2))
    wavesample_vC = (theta_1, v_C0 * np.sin(theta_1 + phi - np.pi/2))
    return wavesample_i, wavesample_v, wavesample_vR, wavesample_vL, wavesample_vC

def phasortips(theta_1, phi, i_0, v_0, v_R0, v_L0, v_C0):
    phasortip_I = (i_0 * np.cos(theta_1 + phi), i_0 * np.sin(theta_1 + phi))
    phasortip_V = (v_0 * np.cos(theta_1), v_0 * np.sin(theta_1))
    phasortip_VR = (v_R0 * np.cos(theta_1 + phi), v_R0 * np.sin(theta_1 + phi))
    phasortip_VL = (v_L0 * np.cos(theta_1 + phi + np.pi/2), v_L0 * np.sin(theta_1 + phi + np.pi/2))
    phasortip_VC = (v_C0 * np.cos(theta_1 + phi - np.pi/2), v_C0 * np.sin(theta_1 + phi - np.pi/2))
    return phasortip_I, phasortip_V, phasortip_VR, phasortip_VL, phasortip_VC

def plot_LCR(theta_1=np.pi/4, i_0=1.0, R=0.5, X_C=0.7, X_L=0.3):
    v_0, v_R0, v_L0, v_C0 = get_peaks_LCR(i_0, R, X_L, X_C)
    ac_case = get_ac_case(X_L, X_C)
    phi = np.arctan((X_C - X_L) / R)
    theta = np.linspace(0, 2*np.pi, 500)
    i, v, v_R, v_L, v_C = calculate_ac(theta, phi, i_0, v_0, v_R0, v_L0, v_C0)
    wavesample = wavesamples(theta_1, phi, i_0, v_0, v_R0, v_L0, v_C0)
    phasortip = phasortips(theta_1, phi, i_0, v_0, v_R0, v_L0, v_C0)
    plot_colors = color_scheme()
    
    fig, ax = setup_canvas()
    ax[1].axhline(0, c='k', lw=1)
    ax[1].axvline(0, c='k', lw=1)
    ax[1].text(max(theta), 0.1, r'$\omega t$', fontsize=10)
    ax[1].set_xlim(0, max(theta))
    ax[1].set_ylim(-1.2, 1.4)
    ax[1].plot(theta, i, c=plot_colors['I'], label=r'$i(t)$')
    ax[1].plot(theta, v, c=plot_colors['V'], label=r'$v(t)$')
    ax[1].plot(theta, v_R, c=plot_colors['V_R'], label=r'$v_R(t)$', alpha=0.4)
    ax[1].plot(theta, v_L, c=plot_colors['V_L'], label=r'$v_L(t)$', alpha=0.4)
    ax[1].plot(theta, v_C, c=plot_colors['V_C'], label=r'$v_C(t)$', alpha=0.4)
    ax[1].legend(frameon=False, loc='upper center', ncol=len(wavesample))
    tickvals = np.linspace(0, 2*np.pi, 5)
    ticklabels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    for idx, tick in enumerate(tickvals):
        ax[1].text(tick-0.15, -0.2, ticklabels[idx], fontsize=8)
        ax[1].plot([tick, tick], [-0.03, 0.03], 'k', lw=0.8)
    ax[1].axvline(theta_1, c='k', lw=1, ls='--', alpha=0.4)
    for idx, ws in enumerate(wavesample):
        if idx < 2:
            ax[1].plot(ws[0], ws[1], 'o', c='gray', markersize=4, alpha=1.0,
                       markeredgecolor='k', markeredgewidth=0.5, zorder=2)
        else:
            ax[1].plot(ws[0], ws[1], 'o', c='white', markersize=4, alpha=0.6,
                       markeredgecolor='k', markeredgewidth=0.5, zorder=2)
    
    ax[0].axhline(0, c='k', lw=1)
    ax[0].axvline(0, c='k', lw=1)
    ax[0].set_xlim(-1.2, 1.2)
    ax[0].set_ylim(-1.2, 1.4)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].plot(i_0 * np.cos(theta), i_0 * np.sin(theta), c=plot_colors['I'], ls='--', alpha=0.4)
    ax[0].plot(v_0 * np.cos(theta), v_0 * np.sin(theta), c=plot_colors['V'], ls='--', alpha=0.4)
    ax[0].plot(v_R0 * np.cos(theta), v_R0 * np.sin(theta), c=plot_colors['V_R'], lw=0.8, ls='--', alpha=0.4)
    ax[0].plot(v_L0 * np.cos(theta), v_L0 * np.sin(theta), c=plot_colors['V_L'], lw=0.8, ls='--', alpha=0.4)
    ax[0].plot(v_C0 * np.cos(theta), v_C0 * np.sin(theta), c=plot_colors['V_C'], lw=0.8, ls='--', alpha=0.4)
    for idx, colorcode in enumerate(['I', 'V', 'V_R', 'V_L', 'V_C']):
        if idx < 2:
            ax[0].arrow(0, 0, phasortip[idx][0], phasortip[idx][1], color=plot_colors[colorcode], lw=1.5,
                        head_width=0.07, head_length=0.1, length_includes_head=True, zorder=0)
        else:
            ax[0].arrow(0, 0, phasortip[idx][0], phasortip[idx][1], color=plot_colors[colorcode], lw=1.5,
                head_width=0.07, head_length=0.1, length_includes_head=True, zorder=1, alpha=0.4)
    for idx, pt in enumerate(phasortip):
        if idx < 2:
            ax[0].plot(pt[0], pt[1], 'o', c='gray', markersize=4, alpha=1.0,
                       markeredgecolor='k', markeredgewidth=0.5, zorder=2)
        else:
            ax[0].plot(pt[0], pt[1], 'o', c='white', markersize=4, alpha=0.6,
                       markeredgecolor='k', markeredgewidth=0.5, zorder=2)
    for idx, colorcode in enumerate(['I', 'V', 'V_R', 'V_L', 'V_C']):
        if idx < 2:
            con_patch = ConnectionPatch(xyA=phasortip[idx], coordsA="data", axesA=ax[0],
                                        xyB=wavesample[idx], axesB=ax[1], color=plot_colors[colorcode], alpha=0.6, zorder=1)
        else:
            con_patch = ConnectionPatch(xyA=phasortip[idx], coordsA="data", axesA=ax[0],
                                        xyB=wavesample[idx], axesB=ax[1], color=plot_colors[colorcode], alpha=0.4, zorder=1)
        fig.add_artist(con_patch)
        
    return fig, ax, ac_case

def plot_LCR_basic(theta_1=np.pi/4, i_0=1.0, R=0.5, X_C=0.7, X_L=0.3):
    v_0, v_R0, v_L0, v_C0 = get_peaks_LCR(i_0, R, X_L, X_C)
    ac_case = get_ac_case(X_L, X_C)
    phi = np.arctan((X_C - X_L) / R)
    theta = np.linspace(0, 2*np.pi, 500)
    i, v, v_R, v_L, v_C = calculate_ac(theta, phi, i_0, v_0, v_R0, v_L0, v_C0)
    wavesample = wavesamples(theta_1, phi, i_0, v_0, v_R0, v_L0, v_C0)
    phasortip = phasortips(theta_1, phi, i_0, v_0, v_R0, v_L0, v_C0)
    plot_colors = color_scheme()
    
    fig, ax = setup_canvas()
    ax[1].axhline(0, c='k', lw=1)
    ax[1].axvline(0, c='k', lw=1)
    ax[1].text(max(theta), 0.1, r'$\omega t$', fontsize=10)
    ax[1].set_xlim(0, max(theta))
    ax[1].set_ylim(-1.2, 1.4)
    ax[1].plot(theta, i, c=plot_colors['I'], label=r'$i(t)$')
    ax[1].plot(theta, v, c=plot_colors['V'], label=r'$v(t)$')
    #ax[1].plot(theta, v_R, c=plot_colors['V_R'], label=r'$v_R(t)$', alpha=0.4)
    #ax[1].plot(theta, v_L, c=plot_colors['V_L'], label=r'$v_L(t)$', alpha=0.4)
    #ax[1].plot(theta, v_C, c=plot_colors['V_C'], label=r'$v_C(t)$', alpha=0.4)
    ax[1].legend(frameon=False, loc='upper center', ncol=2)
    tickvals = np.linspace(0, 2*np.pi, 5)
    ticklabels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    for idx, tick in enumerate(tickvals):
        ax[1].text(tick-0.15, -0.2, ticklabels[idx], fontsize=8)
        ax[1].plot([tick, tick], [-0.03, 0.03], 'k', lw=0.8)
    ax[1].axvline(theta_1, c='k', lw=1, ls='--', alpha=0.4)
    for idx, ws in enumerate(wavesample):
        if idx < 2:
            ax[1].plot(ws[0], ws[1], 'o', c='gray', markersize=4, alpha=1.0,
                       markeredgecolor='k', markeredgewidth=0.5, zorder=2)
        #else:
        #    ax[1].plot(ws[0], ws[1], 'o', c='white', markersize=4, alpha=0.6,
        #               markeredgecolor='k', markeredgewidth=0.5, zorder=2)
    
    ax[0].axhline(0, c='k', lw=1)
    ax[0].axvline(0, c='k', lw=1)
    ax[0].set_xlim(-1.2, 1.2)
    ax[0].set_ylim(-1.2, 1.4)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].plot(i_0 * np.cos(theta), i_0 * np.sin(theta), c=plot_colors['I'], ls='--', alpha=0.4)
    ax[0].plot(v_0 * np.cos(theta), v_0 * np.sin(theta), c=plot_colors['V'], ls='--', alpha=0.4)
    #ax[0].plot(v_R0 * np.cos(theta), v_R0 * np.sin(theta), c=plot_colors['V_R'], lw=0.8, ls='--', alpha=0.4)
    #ax[0].plot(v_L0 * np.cos(theta), v_L0 * np.sin(theta), c=plot_colors['V_L'], lw=0.8, ls='--', alpha=0.4)
    #ax[0].plot(v_C0 * np.cos(theta), v_C0 * np.sin(theta), c=plot_colors['V_C'], lw=0.8, ls='--', alpha=0.4)
    for idx, colorcode in enumerate(['I', 'V', 'V_R', 'V_L', 'V_C']):
        if idx < 2:
            ax[0].arrow(0, 0, phasortip[idx][0], phasortip[idx][1], color=plot_colors[colorcode], lw=1.5,
                        head_width=0.07, head_length=0.1, length_includes_head=True, zorder=0)
        #else:
        #    ax[0].arrow(0, 0, phasortip[idx][0], phasortip[idx][1], color=plot_colors[colorcode], lw=1.5,
        #        head_width=0.07, head_length=0.1, length_includes_head=True, zorder=1, alpha=0.4)
    for idx, pt in enumerate(phasortip):
        if idx < 2:
            ax[0].plot(pt[0], pt[1], 'o', c='gray', markersize=4, alpha=1.0,
                       markeredgecolor='k', markeredgewidth=0.5, zorder=2)
        #else:
        #    ax[0].plot(pt[0], pt[1], 'o', c='white', markersize=4, alpha=0.6,
        #               markeredgecolor='k', markeredgewidth=0.5, zorder=2)
    for idx, colorcode in enumerate(['I', 'V', 'V_R', 'V_L', 'V_C']):
        if idx < 2:
            con_patch = ConnectionPatch(xyA=phasortip[idx], coordsA="data", axesA=ax[0],
                                        xyB=wavesample[idx], axesB=ax[1], color=plot_colors[colorcode], alpha=0.6, zorder=1)
        #else:
        #    con_patch = ConnectionPatch(xyA=phasortip[idx], coordsA="data", axesA=ax[0],
        #                                xyB=wavesample[idx], axesB=ax[1], color=plot_colors[colorcode], alpha=0.4, zorder=1)
        fig.add_artist(con_patch)
        
    return fig, ax, ac_case

def plot_phasor(phi, theta_1=np.pi/4, ac_case="resistive", v_0=1.0, i_0=0.6):
    plot_colors = {'blue':'#0d47a1', 'red':'#da1e37'}
    theta = np.linspace(0, 2*np.pi, 500)
    v = v_0 * np.sin(theta)
    i = i_0 * np.sin(theta + phi)
    
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]}, figsize=(8.5,3), dpi=150)
    fig.tight_layout()
    for pos in ['right', 'top', 'bottom', 'left']:
        ax[0].spines[pos].set_visible(False)
        ax[1].spines[pos].set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    
    wavesample_v = (theta_1, v_0 * np.sin(theta_1))
    wavesample_i = (theta_1, i_0 * np.sin(theta_1 + phi))
    ax[1].axhline(0, c='k', lw=1)
    ax[1].axvline(0, c='k', lw=1)
    ax[1].plot(theta, v, c=plot_colors['blue'], label=r'$v(t)$')
    ax[1].plot(theta, i, c=plot_colors['red'], label=r'$i(t)$')
    ax[1].text(max(theta), 0.1, r'$\omega t$', fontsize=10)
    ax[1].set_xlim(0, max(theta))
    ax[1].set_ylim(-1.2, 1.2)
    ax[1].legend(loc='upper right', fontsize=8)
    ax[1].axvline(theta_1, c='k', lw=1, ls='--', alpha=0.4)
    ax[1].plot(wavesample_v[0], wavesample_v[1], 'o', c=None, markersize=4, alpha=1.0, markeredgecolor='k', markeredgewidth=0.5)
    ax[1].plot(wavesample_i[0], wavesample_i[1], 'o', c=None, markersize=4, alpha=1.0, markeredgecolor='k', markeredgewidth=0.5)
    tickvals = np.linspace(0, 2*np.pi, 5)
    ticklabels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    for idx, tick in enumerate(tickvals):
        ax[1].text(tick-0.15, -0.2, ticklabels[idx], fontsize=8)
        ax[1].plot([tick, tick], [-0.03, 0.03], 'k', lw=0.8)
    
    phasortip_V = (v_0 * np.cos(theta_1), v_0 * np.sin(theta_1))
    phasortip_I = (i_0 * np.cos(theta_1 + phi), i_0 * np.sin(theta_1 + phi))
    ax[0].axhline(0, c='k', lw=1)
    ax[0].axvline(0, c='k', lw=1)
    ax[0].plot(v_0 * np.cos(theta), v_0 * np.sin(theta), 'b--', alpha=0.4)
    ax[0].plot(i_0 * np.cos(theta), i_0 * np.sin(theta), 'r--', alpha=0.4)
    ax[0].plot(phasortip_V[0], phasortip_V[1], 'o', c=None, markersize=4, alpha=1.0, markeredgecolor='k', markeredgewidth=0.5)
    ax[0].plot(phasortip_I[0], phasortip_I[1], 'o', c=None, markersize=4, alpha=1.0, markeredgecolor='k', markeredgewidth=0.5)
    ax[0].arrow(0, 0, phasortip_V[0], phasortip_V[1], color=plot_colors['blue'], lw=1.5,
            head_width=0.07, head_length=0.1, length_includes_head=True, zorder=1)
    ax[0].arrow(0, 0, phasortip_I[0], phasortip_I[1], color=plot_colors['red'], lw=1.5,
            head_width=0.07, head_length=0.1, length_includes_head=True, zorder=0)
    ax[0].set_xlim(-1.2, 1.2)
    ax[0].set_ylim(-1.2, 1.2)
    ax[0].set_aspect('equal', adjustable='box')
    
    con_V = ConnectionPatch(xyA=phasortip_V, coordsA="data", axesA=ax[0],
                      xyB=wavesample_v, axesB=ax[1], color=plot_colors['blue'], alpha=0.6)
    con_I = ConnectionPatch(xyA=phasortip_I, coordsA="data", axesA=ax[0],
                          xyB=wavesample_i, axesB=ax[1], color=plot_colors['red'], alpha=0.6)
    fig.add_artist(con_V)
    fig.add_artist(con_I)
    
    plt.show()
    return fig, ax