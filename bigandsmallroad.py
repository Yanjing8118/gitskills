# -*- coding: UTF-8 -*-
import cv2,os
import numpy as np
from skimage import morphology,data,color
import scipy.ndimage
import rdp
import graph,geom

# city='AerialKITTI'
city='Wuhan'
# city='Bavaria'
image_root='/media/yanjingjing/Seagate Expansion Drive/yanjingjing/dataset/%s-0.5/mask1024/'%city
image_list=list(os.listdir(image_root))
name_list=list(map(lambda x: x[:-4] , image_list))

WIDTH_MAX=100
WIDTH_THRESHOLD=10

def thining(image):

    image = image.astype('float32')/ 255.0
    image[image>0.4]=1.0
    image[image<=0.4]=0
    skeleton =morphology.skeletonize(image)
    skeleton=(skeleton*255).astype('uint8')

    return skeleton

def extract(im, mergin=32):
    # im = scipy.ndimage.imread(in_fname,flatten=True)
    threshold=128
    end1=im.shape[0]-mergin
    end2=im.shape[1]-mergin
    # im2=np.zeros([end1-mergin,end2-mergin,3],dtype='uint8')
    im2= im[mergin:end1,mergin:end2]
    im = np.swapaxes(im2, 0, 1)
    im = im > threshold

    # apply morphological dilation and thinning
    selem = morphology.disk(2)
    im = morphology.binary_dilation(im, selem)
    im = morphology.thin(im)
    im = im.astype('uint8')

    # extract a graph by placing vertices every THRESHOLD pixels, and at all intersections
    vertices = []
    edges = set()
    def add_edge(src, dst):
        if (src, dst) in edges or (dst, src) in edges:
            return
        elif src == dst:
            return
        edges.add((src, dst))
    point_to_neighbors = {}
    q = []
    while True:
        if len(q) > 0:
            lastid, i, j = q.pop()
            path = [vertices[lastid], (i, j)]
            if im[i, j] == 0:
                continue
            point_to_neighbors[(i, j)].remove(lastid)
            if len(point_to_neighbors[(i, j)]) == 0:
                del point_to_neighbors[(i, j)]
        else:
            w = np.where(im > 0)
            if len(w[0]) == 0:
                break
            i, j = w[0][0], w[1][0]
            lastid = len(vertices)
            vertices.append((i, j))
            path = [(i, j)]

        while True:
            im[i, j] = 0
            neighbors = []
            for oi in [-1, 0, 1]:
                for oj in [-1, 0, 1]:
                    ni = i + oi
                    nj = j + oj
                    if ni >= 0 and ni < im.shape[0] and nj >= 0 and nj < im.shape[1] and im[ni, nj] > 0:
                        neighbors.append((ni, nj))
            if len(neighbors) == 1 and (i, j) not in point_to_neighbors:
                ni, nj = neighbors[0]
                path.append((ni, nj))
                i, j = ni, nj
            else:
                if len(path) > 1:
                    path = rdp.rdp(path, 2)
                    if len(path) > 2:
                        for point in path[1:-1]:
                            curid = len(vertices)
                            vertices.append(point)
                            add_edge(lastid, curid)
                            lastid = curid
                    neighbor_count = len(neighbors) + len(point_to_neighbors.get((i, j), []))
                    if neighbor_count == 0 or neighbor_count >= 2:
                        curid = len(vertices)
                        vertices.append(path[-1])
                        add_edge(lastid, curid)
                        lastid = curid
                for ni, nj in neighbors:
                    if (ni, nj) not in point_to_neighbors:
                        point_to_neighbors[(ni, nj)] = set()
                    point_to_neighbors[(ni, nj)].add(lastid)
                    q.append((lastid, ni, nj))
                for neighborid in point_to_neighbors.get((i, j), []):
                    add_edge(neighborid, lastid)
                break
    g = graph.Graph()
    vertex_section = True
    vertices2 = {}
    seen_points = {}
    next_vertex_id=0
    merge_duplicates=False
    for vertex in vertices:
        point = geom.Point(float(vertex[0]+mergin), float(vertex[1]+mergin))
        if point in seen_points and merge_duplicates:
            print('merging duplicate vertex at {}').format(point)
            vertices2[next_vertex_id] = seen_points[point]
        else:
            vertex2 = g.add_vertex(point)
            vertices2[next_vertex_id] = vertex2
            seen_points[point] = vertex2
        next_vertex_id += 1
    for edge in edges:
        src = vertices2[edge[0]]
        dst = vertices2[edge[1]]
        if src == dst and merge_duplicates:
            print('ignoring self edge at {}').format(src.point)
            continue
        g.add_edge(src, dst)
    return g

# def mapextract():
#     imagelist=list(os.listdir(skeleton_root))
#     region_list = list(map(lambda x: x[:-4], imagelist))
#     for region in region_list:
#         in_fname=skeleton_root+region+".png"
#         out_fname=graph_root+region+".graph"
#         extract(in_fname, 128, out_fname,mergin=2)
#         print(region+" OK!")
def get_point2(x,y,d,dlat,dlon):
    x_try=x+d*dlon
    y_try=y+d*dlat
    x_try=int(x_try)
    y_try=int(y_try)
    return x_try,y_try
def get_width(p,mask,ymax,xmax):
    rect=np.zeros((WIDTH_MAX*2,WIDTH_MAX*2),dtype='uint8')
    viz=np.zeros((WIDTH_MAX*2,WIDTH_MAX*2,3),dtype='uint8')
    y=int(p.y)
    x=int(p.x)


    startx=0
    starty=0
    endx=WIDTH_MAX*2-1
    endy=WIDTH_MAX*2-1

    if y<WIDTH_MAX:
        starty=WIDTH_MAX-y
        endy=WIDTH_MAX*2-1
    if y>ymax-WIDTH_MAX:
        starty=0
        endy=WIDTH_MAX+ymax-y-1
    if x<WIDTH_MAX:
        startx=WIDTH_MAX-x
        endx=WIDTH_MAX*2-1
    if x>xmax-WIDTH_MAX:
        startx=0
        endx=WIDTH_MAX+xmax-x-1

    sx=max(x-WIDTH_MAX,0)
    ex=min(x+WIDTH_MAX-1,xmax-1)
    sy=max(y-WIDTH_MAX,0)
    ey=min(y+WIDTH_MAX-1,ymax-1)
    rect[starty:endy,startx:endx]=mask[sy:ey,sx:ex]#裁减出mask

    # cv2.namedWindow("circle", cv2.WINDOW_NORMAL)
    # cv2.imshow("circle",mask)
    # cv2.waitKey(0)
    # cv2.namedWindow("circle", cv2.WINDOW_NORMAL)
    # cv2.imshow("circle",rect)
    # cv2.waitKey(0)


    viz[starty:endy,startx:endx,0]=mask[sy:ey,sx:ex]
    viz[starty:endy,startx:endx,1]=mask[sy:ey,sx:ex]
    viz[starty:endy,startx:endx,2]=mask[sy:ey,sx:ex]

    # 尝试用二分法简化搜索过程
    # 二分法:
    minn = 0               # 最小的下标
    maxx = WIDTH_MAX  # 最大的下标

    rr=0
    while True:
        # allblack=999

        mid = (maxx + minn) // 2 # 中间的下标每次向下取整
        r=mid
        if r==0:
            rr=mid
            break
        x = np.zeros((WIDTH_MAX*2,WIDTH_MAX*2, 1), dtype='uint8')
        # cv2.circle(x,(160,160),r,(255,255,255),-1)
        cv2.circle(x,(WIDTH_MAX,WIDTH_MAX),r,(255,255,255),-1)
        # shixin circle
        # cv2.namedWindow("circle", cv2.WINDOW_NORMAL)
        # cv2.imshow("circle",x)
        # cv2.waitKey(0)

        allblack=999
        for i in range(WIDTH_MAX*2):
            for j in range(WIDTH_MAX*2):
                if x[i,j]==255:
                    if rect[i,j]==255:
                        continue
                    else:
                        allblack=0
                else:
                    continue

        if allblack == 0 :
            minn=0
            maxx=mid
        else:
            rr=mid
            break


    for r in xrange(rr,WIDTH_MAX):
        if r==WIDTH_MAX:
            return r
        x = np.zeros((WIDTH_MAX*2,WIDTH_MAX*2, 1), dtype='uint8')
        # cv2.circle(x,(160,160),r,(255,255,255),-1)
        cv2.circle(x,(WIDTH_MAX,WIDTH_MAX),r,(255,255,255))
        # cv2.namedWindow("circle", cv2.WINDOW_NORMAL)
        # cv2.imshow("circle",x)
        # cv2.waitKey(0)



        for i in range(WIDTH_MAX*2):
            for j in range(WIDTH_MAX*2):
                if x[i,j]==255:
                    if rect[i,j]==255:
                        continue
                    else:
                        # width=r
                        # cv2.circle(viz,(80,80),r,(0,255,0),1)
                        # cv2.namedWindow("viz", cv2.WINDOW_NORMAL)
                        # cv2.imshow("viz",viz)
                        # cv2.waitKey(0)
                        return r
                else:
                    continue


def get_kind_from_rs(current_rs,next_list):
    roadsegment=gc.road_segments[current_rs]
    next_edges=[]
    small=0
    big=0
    node_in_rs=[]

    for edge in roadsegment.edges:
        if edge.src not in node_in_rs:
            node_in_rs.append(edge.src.id)
        if edge.dst not in node_in_rs:
            node_in_rs.append(edge.dst.id)
    for node in node_in_rs:
        if len(g.vertices[node].degree)>=2:
            for next_edge in g.vertices[node].in_edges and g.vertices[node].out_edges:
                if next_edge in roadsegment.edges:
                    pass
                else:
                    next_edges.append(next_edge)
    unsure = True
    if len(next_edges)>0:
        for neighbor in next_edges:
            rss=edge2rs[neighbor.id][0]
            if rs2size[rss]=='big':
                big+=1
                unsure=False
            elif rs2size[rss]=='small':
                small+=1
                unsure=False
            else:
                pass
    else:
        print ('impossible!')
    if small>=big and small!=0:
        kind='small'
        tackled=True
        for next_l in next_list:
            if rs2size[next_l]=='unsure':
                rs2size[next_l]='small'
    if big>small and big!=0:
        tackled=True
        kind='big'
        for next_l in next_list:
            if rs2size[next_l]=='unsure':
                rs2size[next_l]='big'
    if unsure == True:
        for neighbor in next_edges:
            rs=edge2rs[neighbor.id][0]
            if rs in next_list:
                pass
            else:
                next_list.append(rs)
                get_kind_from_rs(rs,next_list)
                if rs2width[current_rs] !='unsure':
                    break

# name_list=['Wuhan_a_1_3_lab','Wuhan_a_1_5_lab','Wuhan_a_4_1_lab','Wuhan_a_4_2_lab']
for name in name_list:
    mask=cv2.imread(image_root+name+'.png',0)
    mask2=cv2.imread(image_root+name+'.png')
    ymax,xmax=mask.shape
    centerline=thining(mask)
    g=extract(centerline,2)


    gc=graph.GraphContainer(g)
    small_edge_list=[]
    big_edge_list=[]
    rs2size={}
    rs2width={}
    edge2rs={}
    for edge in g.edges:
        edge2rs[edge.id]=[]
    for rs in gc.road_segments:
        for edge in rs.edges:
            edge2rs[edge.id].append(rs.id)
    for rs in gc.road_segments:
        node_list=[]
        width_list=[]
        for edge in rs.edges:
            # edge2rs[edge.id].append(rs.id)
            if edge.src not in node_list:
                node_list.append(edge.src)
            if edge.dst not in node_list:
                node_list.append(edge.dst)

        # middle point
        l=len(node_list)
        l2=int(l/2)
        node=node_list[l2]

        Error=True
        iter_num=0
        heading_vector_lon=0
        heading_vector_lat=0
        while Error:
            if len(node.degree)==1:
                loc1=node.point
                loc2=g.vertices[node.degree[0]].point
                dlat = loc1.y - loc2.y
                dlon = loc1.x - loc2.x
                l = np.sqrt(dlat*dlat + dlon * dlon)
                dlat /= l
                dlon /= l
                heading_vector_lat = dlat
                heading_vector_lon = dlon
                Error=False
            elif len(node.degree)==2:
                loc1=g.vertices[node.degree[1]].point
                loc2=g.vertices[node.degree[0]].point
                dlat = loc1.y - loc2.y
                dlon = loc1.x - loc2.x
                l = np.sqrt(dlat*dlat + dlon * dlon)
                dlat /= l
                dlon /= l
                heading_vector_lat = dlat
                heading_vector_lon = dlon
                Error=False
            else:
                print('Error！')
                node=node_list[np.random.randint(0,len(node_list))]
                iter_num+=1
                if iter_num>5:
                    break

        heading_vector_lat_vertical=heading_vector_lon
        heading_vector_lon_vertical=-heading_vector_lat

        if heading_vector_lon * heading_vector_lon + heading_vector_lat * heading_vector_lat < 0.1:
            point=node.point
            width=get_width(point,mask,ymax,xmax)
            width_rs=width
            rs2size[rs.id]='unsure'
            rs2width[rs.id]=width_rs
            continue
        else:
            for d in range(1,512,1):
                x1,y1=get_point2(node.point.x,node.point.y,d,heading_vector_lat_vertical,heading_vector_lon_vertical)
                if x1<0 or y1<0 or x1>=xmax or y1>=ymax:
                    width_rs=d-1
                    rs2width[rs.id]=width_rs
                    print('reach the boundary')
                    break
                elif mask[y1,x1]==0:
                    width_rs=d-1
                    rs2width[rs.id]=width_rs
                    break

        # point=node.point
        # width=get_width(point,mask,ymax,xmax)
        # width_rs=width
        # for node in node_list:
        #     point=node.point
        #     width=get_width(point,mask,ymax,xmax)
        #     width_list.append(width)
        # width_rs=np.average(width_list)
        if width_rs<= WIDTH_THRESHOLD:
            rs2size[rs.id]='small'
            for edge in rs.edges:
                small_edge_list.append([edge,width_rs])
        else:
            rs2size[rs.id]='big'
            for edge in rs.edges:
                big_edge_list.append([edge,width_rs])

    for rs,kind in rs2size.items():
        if kind =='unsure':
            next_rs=[]
            current_rs=rs
            next_rs.append(current_rs)
            get_kind_from_rs(current_rs,next_rs)
            # while kind=='unsure':
            #     roadsegment=gc.road_segments[current_rs]
            #     next_edges=[]
            #     small=0
            #     big=0
            #     node_in_rs=[]
            #
            #     for edge in roadsegment.edges:
            #         if edge.src not in node_in_rs:
            #             node_in_rs.append(edge.src.id)
            #         if edge.dst not in node_in_rs:
            #             node_in_rs.append(edge.dst.id)
            #     for node in node_in_rs:
            #         if len(g.vertices[node].degree)>=2:
            #             for next_edge in g.vertices[node].in_edges and g.vertices[node].out_edges:
            #                 if next_edge in roadsegment.edges:
            #                     pass
            #                 else:
            #                     next_edges.append(next_edge)
            #     unsure = True
            #     if len(next_edges)>0:
            #         for neighbor in next_edges:
            #             rss=edge2rs[neighbor.id][0]
            #             if rs2size[rss]=='big':
            #                 big+=1
            #                 unsure=False
            #             elif rs2size[rss]=='small':
            #                 small+=1
            #                 unsure=False
            #             else:
            #                 pass
            #     else:
            #         print ('impossible!')
            #     if small>=big and small!=0:
            #         kind='small'
            #         rs2size[rs]='small'
            #     if big>small and big!=0:
            #         kind='big'
            #         rs2size[rs]='big'
            #     if unsure == True:
            #         for neighbor in next_edges:
            #             rs=edge2rs[neighbor.id][0]
            kind=rs2size[rs]
            if kind=='small':
                for edge in gc.road_segments[rs].edges:
                    small_edge_list.append([edge,rs2width[rs]])
            if kind=='big':
                for edge in gc.road_segments[rs].edges:
                    big_edge_list.append([edge,rs2width[rs]])





    RGB_road_edges=np.zeros((ymax,xmax,3)).astype(np.uint8)
    # small_road_edges=np.zeros((ymax,xmax,3)).astype(np.uint8)

    for edge,width in small_edge_list:
        n1=edge.src
        n2=edge.dst
        cv2.line(RGB_road_edges,(n1.point.x,n1.point.y),(n2.point.x,n2.point.y),[255,0,0],width*2+2)
    for edge,width in big_edge_list:
        n1=edge.src
        n2=edge.dst
        cv2.line(RGB_road_edges,(n1.point.x,n1.point.y),(n2.point.x,n2.point.y),[103,197,246],width*2+4)
    # kernels1 = np.ones((20, 20))
    # small_road_edges = cv2.dilate(small_road_edges, kernels1)
    # kernels2 = np.ones((50, 50))
    # big_road_edges = cv2.dilate(big_road_edges, kernels2)

    mask2[mask2>0]=RGB_road_edges[mask2>0]
    # mask2[mask2>0]=small_road_edges[mask2>0]+big_road_edges[mask2>0]
    # mask2[mask2>0]=

    cv2.imwrite('/media/yanjingjing/Seagate Expansion Drive/yanjingjing/dataset/%s-0.5/%s.png'%(city,name[:-3]+'rgb'),mask2)




