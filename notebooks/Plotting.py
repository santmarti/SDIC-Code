""" course: System Design Integration and Control SDIC, CSIM, UPF
    contact: marti.sanchez@upf.edu
    Plotting.py : contains plotting auxiliary functions
"""

import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.axis import Tick
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

from Environment import ACT_MODE, OBS_MODE, ACT_PLUS

font = {'family': 'Bitstream Vera Sans', 'size': 20}
x_lim, y_lim = (-4, 4), (-1.5, 5.5)
mycolors, mylabels, myf = [], [], -1


def makeFigure(axes=[], size=[]):
    global x_lim,y_lim

    plt.close()
    if(len(size)>0): fig = plt.figure(figsize=size)
    else: fig = plt.figure()

    ax = plt.axes()
    if(len(axes) > 0):
        x_lim, y_lim = (axes[0], axes[1]), (axes[2], axes[3])
        ax = plt.axes(xlim=x_lim,ylim=y_lim)
        
    ax.set_aspect('equal')
    return fig,ax


def drawCircle(ax, position, r, alpha=0.3, color='b', fill=True, linestyle='solid'):
    lw = 1
    if linestyle is not None:
        c = plt.Circle(position, radius=r, alpha=alpha, fill=fill, facecolor=color, edgecolor=color, ls=linestyle, linewidth=lw)
    else:
        c = plt.Circle(position, facecolor=color, radius=r, alpha=alpha)

    ax.add_patch(c)
    return c

def drawPoly(ax, vertices, alpha=0.5, color = 'b', fill=True, line_style=None, line_width=None):
    if line_width is None:
        lw = 1 if line_style != 'dashed' else 3
    else:
        lw = line_width

    if line_style is None:
        line_style = 'solid'

    poly = plt.Polygon(vertices, alpha=alpha, fill=fill, facecolor=color, edgecolor=color, ls=line_style, linewidth=lw)
    ax.add_patch(poly)


def quatromatrix(visits, left, bottom, right, top, ax=None, cmap='Reds', bNorm=False):
    if not ax: ax=plt.gca()
    n = left.shape[0]; m=left.shape[1]

    a = np.array([[0,0],[0,1],[.5,.5],[1,0],[1,1]])
    tr = np.array([[0,1,2],[0,2,3],[2,3,4],[1,2,4]])

    A = np.zeros((n*m*5,2))
    Tr = np.zeros((n*m*4,3))

    for i in range(n):
        for j in range(m):
            k = i*m+j
            A[k*5:(k+1)*5,:] = np.c_[a[:,0]+j, a[:,1]+i]
            Tr[k*4:(k+1)*4,:] = tr + k*5

    C = np.c_[ left.flatten(), bottom.flatten(), 
              right.flatten(), top.flatten() ].flatten()

    vf = visits
    Cv = np.c_[ vf.flatten(), vf.flatten(),
              vf.flatten(), vf.flatten() ].flatten()

    plt.pcolor(0.5*visits, cmap='Greys', alpha=0.5)
    norm = None if not bNorm else colors.Normalize(vmin=0, vmax=1, clip=True)
    ax.triplot(A[:, 0], A[:, 1][::-1], Tr, mask=Cv,  color=[.7,.7,.7], alpha=0.9)
    tripcolor = ax.tripcolor(A[:, 0], A[:, 1][::-1], Tr, mask=Cv, facecolors=C, cmap=cmap, alpha=0.9, norm=norm)
    ax.margins(0)
    ax.grid(True)
    return tripcolor

def plotQ(env, q, mask_visits=False, min_max=None):
    nactions = env.nA
    nstates = env.nS
    state_shape = env.my_env.grid_shape
    #print("Shape of state space: " + str(state_shape))

    qmat = [np.zeros(state_shape) for i in range(nactions)]

    visits = np.zeros(state_shape)

    for s in range(nstates): 
        i,j = env.index2state(s)

        if(mask_visits):
            visits[i,j] = 0 if q.visited(s) else 1

        actions =  [3,0,1,2] if env.name.startswith("CliffWalking") else [3,2,1,0]   # Actions of cliff environment: UP-0,RIGHT-1,DOWN-2,LEFT-3
        for ai,a in enumerate(actions):
            aq = q.predict(s)[a]
            if min_max is not None:
                aq = np.clip(aq, min_max[0], min_max[1])
            qmat[ai][i,j] = aq

    fig = plt.figure(figsize=(20, 5))
    ax = plt.axes()
    ax.set_aspect("equal")

    # Actions of cliff environment: UP-0,RIGHT-1,DOWN-2,LEFT-3
    # Plot function wants: left, bottom, right, top 
    tri_plot = quatromatrix(visits, qmat[0], qmat[1], qmat[2], qmat[3], ax=ax, cmap='viridis')
    
    #ax.margins(0)
    ax.grid(False)

    fig.colorbar(tri_plot)
    plt.show()


def matrix_plot(a, size=(8,5), colorbar=True):
    fig = plt.figure(figsize=size)
    ax = plt.axes()
    pc = ax.pcolor(np.flip(a, axis=0))
    ax.set_aspect("equal")
    if colorbar:
        fig.colorbar(pc, ax=ax)
    return fig,ax


def decorate(xlabel = None, ylabel = None, title = None, xticks = None, mainfont=20, legendfont=12, bLegend = False):
    global font
    font['size']=mainfont
    plt.rc('font', **font)
    plt.rc('legend',**{'fontsize':legendfont})
    if(xlabel != None): plt.xlabel(xlabel)
    if(ylabel != None): plt.ylabel(ylabel)
    if(title != None): plt.title(title)
    if(bLegend): plt.legend()


def vrotate(v, angle, anchor=[0,0], mag=1):
    """Rotate a vector `v` by the given angle, relative to the anchor point."""
    x, y = v
    x = x - anchor[0]
    y = y - anchor[1]
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    nx = x*cos_theta - y*sin_theta
    ny = x*sin_theta + y*cos_theta
    nx = nx + anchor[0]
    ny = ny + anchor[1]
    return [mag*nx, mag*ny]

def computePointsAngle(pos, vdir, dx=1):
    if(np.count_nonzero(np.array(vdir)) > 0):
        vdir90 = vrotate(vdir, np.pi/2)
        p1 = [pos[0]+dx*vdir90[0],pos[1]+dx*vdir90[1]]
        p2 = [pos[0]-dx*vdir90[0],pos[1]-dx*vdir90[1]]
        p3 = [pos[0]+2.8*dx*vdir[0],pos[1]+2.8*dx*vdir[1]]
        return [p1, p2, p3]
    else:
        p1 = [pos[0]+dx,pos[1]+dx]
        p2 = [pos[0]+dx,pos[1]-dx]
        p3 = [pos[0]-dx,pos[1]-dx]
        p4 = [pos[0]-dx,pos[1]+dx]
        return [p1, p2, p3, p4]




def decorate(xlabel = None, ylabel = None, title = None, xticks = None, mainfont=20, legendfont=12, bLegend = False):
    global font
    font['size']=mainfont
    plt.rc('font', **font)
    plt.rc('legend',**{'fontsize':legendfont})
    if(xlabel != None): plt.xlabel(xlabel)
    if(ylabel != None): plt.ylabel(ylabel)
    if(title != None): plt.title(title)
    if(bLegend): plt.legend()



def drawBarPlot(values, xlabel = None, ylabel = None, title = None, xticks = None, color = 'b'):
    plt.rc('font', **font)
    N = len(values)
    maxv = max(values)
    index = np.arange(N)  # the x locations for the groups
    barWidth = 0.35       # the width of the bars
    fig, ax = makeFigure(xlim=(-barWidth, N - barWidth/2), ylim=(0, maxv + maxv/5.0))
    ax.bar(index, values, barWidth, color=color, error_kw=dict(elinewidth=2,ecolor='red'))
    if(xticks != None): plt.xticks(index + barWidth/2.0, xticks)
    decorate(xlabel,ylabel,title,xticks)


def drawPlotY(y, xlim = [0,100], label = None, color = None, linewidth=2, colors = [], labels = [], line='-'):
    global mycolors,mylabels,myf
    if(color != None or label != None): myf = -1
    if(len(colors) > 0): mycolors, myf = colors, 0
    if(len(labels) > 0): mylabels, myf = labels, 0

    x = np.linspace(xlim[0], xlim[1], len(y))

    if(myf < 0):
        if(color != None): p, = plt.plot(x,y,color=color,label=label)
        else: p, = plt.plot(x,y)
    else:
        p, = plt.plot(x,y,line,color=mycolors[myf],label=mylabels[myf])
        myf += 1

    plt.setp(p, linewidth=linewidth)

    return p

def drawPlotXY(x, y, yerror = None, xlabel = None, ylabel = None, title = None, xticks = None, color = None, myplt = None):
    if(myplt == None): myplt = plt     
    if(color != None): p, = myplt.plot(x,y,color=color)
    else: p, = myplt.plot(x,y)
    #if(yerror != None): plt.fill_between(x, y-yerror, y+yerror, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    if(yerror != None): 
        yminus = np.array(y)-np.array(yerror)
        yplus = np.array(y)+np.array(yerror)
        myplt.fill_between(x, yminus, yplus, alpha=0.5, linewidth=0, color=color)
    decorate(xlabel,ylabel,title,xticks)
    return p



def clear_plt():
    plt.clf()
    ax = plt.axes()
    ax.set_aspect('equal')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.xticks([])
    plt.yticks([])
    return ax


def draw_agent(xy, state, conf, ax, action=None, params={}):
    x, y = xy
    if type(state) in [int, np.int64, np.int32]:
        orientation = state
        nx, ny = conf["rows"], conf["cols"]
    else:
        a_key = list(state.keys())[0]
        nx, ny = state[a_key].shape
        orientation = int(state["agents"][int(x), int(y)]) if "agents" in state else 1

    num_agents = 1 if "num_agents" not in conf else conf["num_agents"]
    a_mode = ACT_MODE.ALLOCENTRIC if "action_mode" not in conf else conf["action_mode"]

    agent_color = None if not "agent_color" in params else params["agent_color"]
    agent_alpha = 0.65 if not "agent_alpha" in params else params["agent_alpha"]

    bPlotObs, acolor, minus_color = False, [.4, .1, .15], [.4, -.1, -.15]
    if action is not None:
        if action >= 3 and a_mode is ACT_MODE.EGOCENTRIC:
            if num_agents == 1:
                bPlotObs = True
            acolor = [0, 0, .6]
        if action >= 4 and a_mode is ACT_MODE.ALLOCENTRIC:
            if num_agents == 1:
                bPlotObs = True
            acolor = [0, 0, .6]
        if "action_plus" in conf and conf["action_plus"] is ACT_PLUS.NOTHING_RANDOM_HARVEST and action >= 2:
            if num_agents == 1:
                bPlotObs = True
            acolor = [0, 0, .6]

    bPlotObs_Prio = True if "plot_obs" not in params else params["plot_obs"]
    bPlotObs_Prio = bPlotObs_Prio or (action is not None and action > 2)

    incr_obs = 0
    if "action_plus" in conf and conf["action_plus"] is ACT_PLUS.NOTHING_OBS_RADIUS:
        r_plus = conf["obs_radius_plus"] if "obs_radius_plus" in conf else 1
        if action == 3 and a_mode is ACT_MODE.EGOCENTRIC: incr_obs = r_plus
        if action == 4 and a_mode is ACT_MODE.ALLOCENTRIC: incr_obs = r_plus

    # To plot the observation radius field
    if "obs_mode" in conf and conf["obs_mode"] not in [OBS_MODE.GLOBAL, OBS_MODE.GLOBAL_CENTER_PAD, OBS_MODE.GLOBAL_CENTER_WRAP]:
        obs_r = 1 if "obs_radius" not in conf else conf["obs_radius"]
        obs_r += incr_obs

        if bPlotObs or num_agents < 2 or (num_agents < 5 and obs_r < 3 and nx > 5) or nx > 30:
            if bPlotObs_Prio:
                dx = 7.9*obs_r*0.15
                if obs_r == 1: dx += 0.2
                vertices_radius = computePointsAngle([y,nx-x-1],[0,0], dx=dx)
                vertices_radius += np.array([0.5,0.5])

                cobs = np.clip(np.array(acolor) + np.array(minus_color), 0, 1)
                aobs = 0.1 if num_agents < 2 else 0.05
                drawPoly(ax, vertices_radius, alpha=aobs, color=cobs, line_style='dashed')

                if "torus" in conf and conf["torus"]:
                    for sx,sy in [[nx,ny], [nx,0], [0,ny], [-nx,0], [0,-ny], [-nx,-ny], [-nx,ny], [nx,-ny]]:
                        vertices_radius = computePointsAngle([y-sy,nx-x-1-sx],[0,0], dx=dx)
                        vertices_radius += np.array([0.5,0.5])
                        drawPoly(ax, vertices_radius, alpha=aobs, color=cobs, line_style='dashed')


    if a_mode is ACT_MODE.ALLOCENTRIC:
        vertices = computePointsAngle([y, nx-x-1], [0, 0], dx=0.2)
        vertices -= np.array([-0.5, -0.5])

    if a_mode is ACT_MODE.EGOCENTRIC:
        dirs = [[0, 1], [-1, 0], [0, -1], [1, 0]]

        #v = dirs[ (orientation-1+4)%4 ]
        v = dirs[orientation-1]
        mydir = [v[1], -v[0]]
        centered_dir = .2 if "centered_dir" not in params else params["centered_dir"]
        asize = 0.15 if "agent_size" not in params else params["agent_size"]

        #v = dirs[ (orientation-2+4)%4 ]
        v = dirs[orientation-2]
        vertices = computePointsAngle([y-centered_dir*v[0], nx-x-1-centered_dir*v[1]], mydir, dx=asize)
        vertices -= np.array([-0.5, -0.5])

    if agent_color is not None:
        acolor = agent_color

    line_width = None if not "line_width" in params else params["line_width"]
    line_style = None if not "line_style" in params else params["line_style"]
    drawPoly(ax, vertices, color=acolor, alpha=agent_alpha, line_style='solid', line_width=line_width)



def draw_agents(state, conf, ax, agents_xy=None, actions=None, params={}):
    bPlotAgents = params["agents"] if "agents" in params else True
    if not bPlotAgents:
        return

    if "agents" in state:
        if agents_xy is None:
            xs, ys = np.nonzero(state["agents"])
            agents_xy = zip(xs,ys)

        for i, xy in enumerate(agents_xy):
            if actions is not None:
                draw_agent(xy, state, conf, ax, action=actions[i], params=params)
            else:
                draw_agent(xy, state, conf, ax, params=params)



def plot_icon(ax, xy, icon_fname='goal_icon.png', zoom=0.05, dx=0):
    g_icon = mpimg.imread("./icons/" + icon_fname)
    imagebox = OffsetImage(g_icon, zoom=zoom)
    ab = AnnotationBbox(imagebox, (xy[0]+.5+dx, xy[1]+.5), frameon=False)
    ax.add_artist(ab)



def draw_obs(env, obs, ax=None, params={}, enemy_info=[], save=None):
    if save is not None:
        fig = plt.figure(figsize=(3,3))        
    if ax is None:
        ax = clear_plt()

    conf = env.conf
    num_agents = 1 if "num_agents" not in conf else conf["num_agents"]
    layer_names = env.get_state_map().keys()

    if num_agents == 1:
        layer_names = list(layer_names - ["agents"])

    ox, oy = obs[0].shape

    state = {}
    for i, l in enumerate(layer_names):
        state[l] = obs[i]

    plt.xticks(range(oy + 1))  # for plotting x axis corresponds to cols
    plt.yticks(range(ox + 1))  # and y axis corresponds to rows, and counting starts at 0 so we need one more
    plt.grid(True)

    plotted_layer = draw_walls(state, params)
    plotted_layer = draw_floor(state, conf, ax, plotted_layer, params)
    plotted_layer = draw_objects(state, ax, plotted_layer, params)
    plotted_layer = draw_enemies(state, conf, ax, plotted_layer, params, enemy_info=enemy_info)
    draw_food(state, ax, plotted_layer, params)

    bAddedAgents = False
    xy = [int(ox/2), int(oy/2)]
    if "agents" not in state:
        bAddedAgents = True
        state["agents"] = np.zeros((ox, oy))
    state["agents"][xy[0], xy[1]] = 1

    extra_params = params.copy()
    extra_params["agent_alpha"] = 0.2
    extra_params["line_width"] = 1.5
    extra_params["line_style"] = "dashed"

    draw_agent(xy, state, conf, ax=ax, params=extra_params)

    if bAddedAgents:
        del state["agents"]
    else:
        state["agents"][xy[0], xy[1]] = 0

    if num_agents > 1:
        params["plot_obs"] = False
        draw_agents(state, conf, ax, params=params)

    if "no_ticks" not in params:
        plt.xticks(range(oy + 1))  # for plotting x axis corresponds to cols
        plt.yticks(range(ox + 1))  # and y axis corresponds to rows, and counting starts at 0 so we need one more
    plt.grid(True)

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches='tight')    



def draw_walls(state, plotted_layer=False, params={}):
    bPlotWalls = params["walls"] if "walls" in params else True
    if ("walls" in state) and bPlotWalls:
        alpha_walls = .6 if "walls_alpha" not in params else params["walls_alpha"]
        plt.pcolor(np.flip(state["walls"], axis=0), cmap='Greys', alpha=alpha_walls)
        plotted_layer = True
    return plotted_layer


def draw_objects(state, ax=None, plotted_layer=False, params={}):
    bPlotObj = params["objects"] if "objects" in params else True
    if "objects" in state and bPlotObj:
        nx, ny = state["objects"].shape
        if not plotted_layer:
            zeros = np.zeros_like(state["objects"])
            plt.pcolor(zeros, cmap='Greys')
            plotted_layer = True

        mycmap = ["w", "b", "g", "r"]
        xs, ys = np.nonzero(state["objects"])
        for x, y in zip(xs, ys):
            o = int(state["objects"][x, y])
            r = 0.2 if state["agents"][int(x), int(y)] == 0 else 0.15
            drawCircle(ax, [y + 0.5, nx - x + 0.5 - 1], r=r, color=mycmap[o])
    return plotted_layer


def draw_food(state, ax=None, plotted_layer=False, params={}):
    bPlotFood = params["food"] if "food" in params else True
    if "food" in state and bPlotFood:
        nx, ny = state["food"].shape
        if not plotted_layer:
            zeros = np.zeros_like(state["food"])
            plt.pcolor(zeros, cmap='Greys')
            plotted_layer = True
        xs, ys = np.nonzero(state["food"])
        for x, y in zip(xs, ys):
            drawCircle(ax, [y + 0.5, nx - x + 0.5 - 1], r=0.21, color='g')
    return plotted_layer

def draw_enemies(state, conf, ax=None, plotted_layer=False, params={}, enemy_info=[]):
    bPlotEne = params["enemies"] if "enemies" in params else True

    if "enemies" in state and bPlotEne:
        nx, ny = state["enemies"].shape

        if not plotted_layer:
            zeros = np.zeros_like(state["enemies"])
            plt.pcolor(zeros, cmap="Greys")
            plotted_layer = True

        for enemy in enemy_info:
            x,y=enemy["position"]
            vertices = computePointsAngle([y, nx - x - 1], [0, 0], dx = 0.25)
            vertices -= np.array([-0.5, -0.5])
            drawPoly(ax, vertices, color="Black", alpha=.3)

            if enemy["attack_range"] > 0: #Draw attack area
                dx = 7.9 * enemy["attack_range"] * 0.15
                if enemy["attack_range"] == 1: dx += 0.2
                vertices_radius = computePointsAngle([y, nx - x - 1], [0, 0], dx=dx)
                vertices_radius += np.array([0.5, 0.5])
                # clip vertices to not overcome the environment limits
                vertices_radius[:,0] = np.clip(vertices_radius[:,0], 0.115, ny - 0.115)
                vertices_radius[:,1] = np.clip(vertices_radius[:,1], 0.115, nx - 0.115)
                drawPoly(ax, vertices_radius, alpha=90, color="LightGray", line_style='solid')
                if "torus" in conf and conf["torus"]:
                    for sx,sy in [[nx,ny], [nx,0], [0,ny], [-nx,0], [0,-ny], [-nx,-ny], [-nx,ny], [nx,-ny]]:
                        vertices_radius = computePointsAngle([y-sy,nx-x- 1-sx],[0,0], dx=dx)
                        vertices_radius += np.array([0.5,0.5])
                        drawPoly(ax, vertices_radius, alpha=90, color="LightGray", line_style='solid')

    return plotted_layer

def draw_floor(state, conf, ax=None, plotted_layer=False, params={}):
    bPlotFloor = params["floor"] if "floor" in params else True
    if "floor" in state and bPlotFloor:
        nx, ny = state["floor"].shape
        cmap, vmax, alpha = 'GnBu', 2, 0.3

        if "update" in conf:
            vmax, cmap= 3, colors.ListedColormap(["w", [0.2, .7, 0], [1, 0, 0.2], [0, 0.3, 0.3]])
            if conf["update"] is "forest_fire":
                vmax, cmap = 3, colors.ListedColormap([[.9, .9, .9], [0, .7, 0.1], [0.7, 0.0, 0.1], [0, 0.3, 0.3]])
        if "areas" in conf["floor"]:
            vmax, cmap = 4, colors.ListedColormap(["w", [.0, .0, .7], [.7, .1, .1], [.2, .9, .2], [.8, .9, 0]])
            alpha = 0.3
        elif "blocks" in conf["floor"]:
            vmax, cmap = 4, colors.ListedColormap(["w", "b", "g", "r"])
            alpha = 0.3
        elif "goal_sequence" in conf["floor"]:
            vmax, cmap = 3, colors.ListedColormap(["w", "g", "b", "r"])
        elif "harlow" in conf["floor"]:
            vmax, cmap = 4, colors.ListedColormap(["w", "g", "b", "g", "r"])
        elif "object_areas" in conf["floor"]:
            vmax, cmap = 3, colors.ListedColormap(["w", "b", "g", "r"])
            alpha = 0.2

        alpha = alpha if "floor_alpha" not in params else params["floor_alpha"]
        state_int = np.flip(state["floor"], axis=0).astype(int)
        plt.pcolor(state_int, vmin=0, vmax=vmax, cmap=cmap, alpha=min(1, alpha))
        plotted_layer = True
        zoom_icon = 0.05 if "zoom_icon" not in params else params["zoom_icon"]
        floor_max = np.max(state["floor"])
        if floor_max > 5:
            # change condition / use params
            xs = np.where(state["floor"] == floor_max)[0]
            ys = np.where(state["floor"] == floor_max)[1]
            for i,j in zip(xs,ys):
                plot_icon(ax, [j,nx - i - 1], zoom=zoom_icon)
            n = 1
            cell_alpha = 0.15 if "cell_alpha" not in params else params["cell_alpha"]
            xs = np.where(state["floor"]+state["walls"] == 0)[0]
            ys = np.where(state["floor"]+state["walls"] == 0)[1]
            if len(xs) < 3:
                for i,j in zip(xs,ys):
                    #plot_icon(ax, [j,i], icon_fname="arrow_icon.png", zoom=0.1, dx=-0.2)
                    drawCircle(ax, [j + 0.5, nx - i + 0.5 - 1], r=.4, color='g', alpha=cell_alpha, fill=True, linestyle=None)
                    bPlotText = params["text"] if "text" in params else True
                    if bPlotText:
                        fsize = 15 * (1 - 10*(0.05 - zoom_icon))
                        ax.text(j + 0.5, nx - i + 0.5 - 1, str(n), alpha=cell_alpha+0.2, fontsize=fsize, color=[0,0.3,0], verticalalignment='center', horizontalalignment='center')
                    n += 1
    return plotted_layer


def draw_layers(conf, state, agents_xy=None, actions=None, ax=None, params={}, enemy_info=[]):
    if ax is None:
        ax = clear_plt()

    plotted_layer = draw_walls(state, params=params)
    plotted_layer = draw_floor(state, conf, ax, plotted_layer, params=params)
    plotted_layer = draw_objects(state, ax, plotted_layer, params=params)
    plotted_layer = draw_enemies(state, conf, ax, plotted_layer, params, enemy_info=enemy_info)
    draw_food(state, ax, plotted_layer, params=params)

    nx, ny = conf["rows"], conf["cols"]

    bGrid = params["grid"] if "grid" in params else True
    if bGrid and nx < 50:
        plt.xticks(range(ny + 1))  # for plotting x axis corresponds to cols
        plt.yticks(range(nx + 1))  # and y axis corresponds to rows, and counting starts at 0 so we need one more
        plt.grid(True)
        for xtic in ax.xaxis.get_major_ticks():
            xtic.tick1line.set_visible(False)
        for ytic in ax.yaxis.get_major_ticks():
            ytic.tick1line.set_visible(False)
    draw_agents(state, conf, ax, agents_xy, actions, params=params)

def plotBetweenMinMax(data_mean, data_min, data_max, mycolor=[1,0,0,1], mylabel='', linewidth=1, alpha=0.15):
    x = np.arange(len(data_mean))
    plt.fill_between(x, data_min, data_max, alpha=alpha, linewidth=0, color=mycolor)
    plt.plot(data_mean, c=mycolor, ls='-', label=mylabel, linewidth=linewidth)

def plotBetween(data_mean, data_std, mycolor=[1,0,0,1], mylabel='', linewidth=1, alpha=0.15):
    data_mean_minus = data_mean - data_std 
    data_mean_plus = data_mean + data_std
    x = np.arange(len(data_mean))
    plt.fill_between(x, data_mean_minus, data_mean_plus, alpha=alpha, linewidth=0, color=mycolor)
    plt.plot(data_mean, c=mycolor, ls='-', label=mylabel, linewidth=linewidth)

def plotMeanStd(data_vec, mycolor=[1,0,0,1], mylabel='', linewidth=1, alpha=0.15):           
    data_vec = np.array(data_vec)             # length number of trials
    data_mean = np.mean(data_vec, axis=0)     # length number of timesteps
    data_std = np.std(data_vec, axis=0)
    plotBetween(data_mean, data_std,  mycolor=mycolor, mylabel=mylabel, linewidth=linewidth, alpha=alpha)
    return len(data_mean)

def plotMeanMinMax(data_vec, mycolor=[1,0,0,1], mylabel='', linewidth=1, alpha=0.15):
    data_vec = np.array(data_vec)             # length number of trials
    data_mean = np.mean(data_vec, axis=0)     # length number of timesteps
    data_min = np.min(data_vec, axis=0)
    data_max = np.max(data_vec, axis=0)
    plotBetweenMinMax(data_mean, data_min,  data_max, mycolor=mycolor, mylabel=mylabel, linewidth=linewidth, alpha=alpha)
    return len(data_mean)


def plotRunningWindow(data_vec, window=10, mycolor=[1,0,0,1], mylabel='', linewidth=1, alpha=0.15):
    data_vec = np.array(data_vec)
    data_mean = np.array([data_vec[i:i + window].mean() for i in range(data_vec.size - window)])
    data_std = np.array([data_vec[i:i + window].std() for i in range(data_vec.size - window)])
    plotBetween(data_mean, data_std, mycolor=mycolor, mylabel=mylabel, linewidth=linewidth, alpha=alpha)


def plotMatrix(M, axis_ticks=None, axis_labels=None, lim=[None, None], bColorBar=True, bColorBarOnly=False, cbar_params=[0.035, 0.05], fig=None):
    ax = plt.gca()
    cplot = plt.pcolor(M.getM(), vmin=lim[0], vmax=lim[1], cmap='Greys')
    if bColorBar:
        if len(cbar_params) > 2:
            cbar_axes = fig.add_axes(cbar_params)
            plt.colorbar(cplot, cax=cbar_axes)
        else:
            plt.colorbar(cplot, fraction=cbar_params[0], pad=cbar_params[1])

    if bColorBarOnly:
        ax.remove()
        return

    xlim, ylim = M.xlim, M.ylim
    ax.set_aspect('equal')

    x_labels = ["%.2f" % l for l in np.linspace(xlim[0], xlim[1], num=len(ax.get_xticklabels()))]
    y_labels = ["%.4f" % l for l in np.linspace(ylim[0], ylim[1], num=len(ax.get_yticklabels()))]
    if axis_ticks is not None:
        if not axis_ticks[0]:
            x_labels = []
            ax.set_xticklabels(x_labels)
        if not axis_ticks[1]:
            y_labels = []
            ax.set_yticklabels(y_labels)

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    if axis_labels is None:
        plt.xlabel('Prob of tree')
        plt.ylabel('Prob of fire')
    else:
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])



def plot(td_list, ax=None, color=None, label=None, title=None, xlabel='Updates', ylabel='Error', ylim=None, marker=None):
    if ax == None:
        _, ax = plt.subplots()
    plt.plot(td_list, label=label, color=color, marker=marker)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title: plt.title(title)
    if label: plt.legend()
    if ylim is not None: ax.set_ylim(ylim)
    return ax