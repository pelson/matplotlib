
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.transforms as mtrans
import matplotlib.patches as mpatches


def main1():
    plt.figure()
    
    ax = plt.axes()
    ax.set_xlim([-18, 20])
    ax.set_ylim([-150, 100])
    
    path = mpath.Path.unit_regular_star(8)
    path.vertices *= [10, 100]
    path.vertices -= [5, 25]
    
    path2 = mpath.Path.unit_circle()
    path2.vertices *= [10, 100]
    path2.vertices += [10, -25]
    
    combined = mpath.Path.make_compound_path(path, path2)
    
    patch = mpatches.PathPatch(
        combined, alpha=0.5, facecolor='coral', edgecolor='none')
    ax.add_patch(patch)
    
    bbox = mtrans.Bbox([[-12, -77.5], [50, -110]])
    result_path = combined.clip_to_bbox(bbox)
    result_patch = mpatches.PathPatch(
        result_path, alpha=0.5, facecolor='green', lw=4, edgecolor='black')
    #ax.add_patch(result_patch)
    
    x0, y0, x1, y1 = -12, -77.5, 50, -110
    bbox_path = mpath.Path([[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]])
    r = combined.clip_line_to_bbox(bbox_path)
    
    #print 'RESULT:', r
    
    #plt.show()
    
def main2():
    
    import numpy as np
    import shapely.geometry as sgeom
    
    shp = sgeom.polygon.LinearRing([[170, 0.0], [175, 0], [190, 0]])
    coords = np.array(shp.coords)
    
    clip_coords = np.array([[0, -90], [180, -90], [180, 90], [0, 90], [0, -90]])
    
    p = mpath.Path(coords)
    clip_p = mpath.Path(clip_coords)
    
    r = p.clip_line_to_bbox(clip_p)
    print r
    
if __name__ == '__main__':
    main2()