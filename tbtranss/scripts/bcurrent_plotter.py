import argparse
import numpy as np
from mayavi import mlab

parser = argparse.ArgumentParser(description="Plot bond currents from the '*.npz' files created by ScattererBuilder "
                                 "(geometrical information) and GreenSystem (bond current information).")

parser.add_argument("-g", "--geometry", type=str, nargs=1, help="File path of the required geometry file produced by "
                    "ScattererBuilder.")
parser.add_argument("-b", "--bcurrent", type=str, nargs=1, help="File path of the required bond current file produced "
                    "by GreenSystem.")
parser.add_argument("-t", "--title", type=str, nargs=1, help="Title of the plot output.")
args = parser.parse_args()
param = vars(args)


title = param["title"][0]
geofile = np.load(param["geometry"][0])
currentfile = np.load(param["bcurrent"][0])


bcurrent = currentfile["bcurrent"]
idxmap = geofile["idxmap"]
P = geofile["conmap"]
N = np.hstack((P[:, :3] + P[:, 3:], -P[:, 3:]))

buff = np.where(idxmap)
currentlist = bcurrent[np.where(idxmap)]
currentlist = np.array(currentlist)
colormap = np.abs(currentlist)
poscheck = np.greater(currentlist, 0)
negcheck = np.less(currentlist, 0)
posmap = colormap[poscheck]
negmap = colormap[negcheck]

cmax = np.max(colormap)
cmin = 0


P = P[poscheck]
pts_pos = mlab.quiver3d(P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4], P[:, 5], scalars=posmap, colormap="Reds",
                        scale_factor=1, vmin=cmin, vmax=cmax)
pts_pos.glyph.color_mode = "color_by_scalar"

N = N[negcheck]
pts_neg = mlab.quiver3d(N[:, 0], N[:, 1], N[:, 2], N[:, 3], N[:, 4], N[:, 5], scalars=negmap, colormap="Reds",
                        scale_factor=1, vmin=cmin, vmax=cmax)
pts_neg.glyph.color_mode = "color_by_scalar"

mlab.title(title)
mlab.scalarbar()

mlab.show()
