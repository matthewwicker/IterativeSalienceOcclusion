# Author: Matthew Wicker
# Contact: matthew.wicker@cs.ox.ac.uk

import numpy as np
import sys
import copy

    
def ISO_vox(inp, label, model, trials=1, r_passes=1, retocc=False):
    inp = np.squeeze(inp)
    model_in = copy.deepcopy(inp)
    in_adver = copy.deepcopy(inp)
    true_lbl, true_lbl_num = label, np.argmax(label)
    
    initial_pred = predict(model_in, model)
    if(np.argmax(initial_pred) != true_lbl_num):
        return 0
    
    latent = get_max_pool(model, model_in)
    saliency_values = [] # this calculates the saliency of all values in the voxelization
    already_counted_indexes = []
    """
    okay so what is happening here is we are taking every point 
    and we give it t
    """
    x,y,z = np.nonzero(inp)
    list_ones = zip(x,y,z)
    
    for i in list_ones: 
        a,b,c = i
        assert(inp[a][b][c] == 1)
        x,y,z=a/5.4,b/5.4,c/5.4
        if(not([int(i[0]), int(i[1]), int(i[2])] in already_counted_indexes)):
            already_counted_indexes.append([int(x), int(y), int(z)])
            saliency_values.append(max(latent[0][int(x)][int(y)][int(z)]))
    saliency_values += 2*abs(min(saliency_values))
    saliency_values /= sum(saliency_values)
    
    manipulations = []
    inds_changed = []
    size = len(list_ones)
    while(True):
        #Make a manipulation
        ind = np.random.choice(range(len(saliency_values)), p=saliency_values)
        i,j,k = list_ones[ind]
        if(in_adver[i][j][k] == 1.0):
            in_adver[i][j][k] = 0.0
            manipulations.append([i,j,k])
            conf, cl = predict(in_adver, model)
            sys.stdout.write("Points occluded: %s conf: %s \r"%(len(manipulations), conf))
            sys.stdout.flush()
            if(cl != true_lbl_num):
                break
            #Update saliency values
            saliency_values[ind] = 0
            saliency_values/=sum(saliency_values)
            #x,y,z = np.nonzero(in_adver)
            #list_ones = zip(x,y,z)
            #saliency_values = []
            #for i in list_ones: 
            #    a,b,c = i
            #    x,y,z=a/5.4,b/5.4,c/5.4
            #    if(not([int(i[0]), int(i[1]), int(i[2])] in already_counted_indexes)):
            #        already_counted_indexes.append([int(x), int(y), int(z)])
            #        saliency_values.append(max(latent[0][int(x)][int(y)][int(z)]))
            #saliency_values += 2*abs(min(saliency_values))
            #saliency_values /= sum(saliency_values)
        else:
            assert(False)
        
    for _ in range(r_passes):
        indices = []
        ind_value = 0
        for i in manipulations:
            in_adver[i[0]][i[1]][i[2]] = 1.0
            conf, cl = predict(in_adver, model)
            if(cl != true_lbl_num):
                indices.append(ind_value)
            else:
                in_adver[i[0]][i[1]][i[2]] = 0.0
            ind_value += 1
        for i in sorted(indices, reverse=True):
            del manipulations[i]
            
    return len(manipulations), manipulations, in_adver
    
    
    
    
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.min(dist_2) #argmin will give us the index, right now we want the l2 distance

def ISO_point(inp, label, model, monotone=True):
    confidences = [1]
    removed = []
    removed_ind = []
    points_occluded = 0
    x = list(inp)
    y = np.argmax(label)
    
    # First you have to calculate an all pairs distance
    # and keep a matrix of points with the smallest change
    nearest_neighbors = []
    for i in range(len(x)):
        replace = x[0]
        del x[0]
        nearest_neighbors.append(closest_node(replace, x))
        x.append(replace)
        
    conf_i, cl = predict(x,model)
    if(cl != y):
        return 0,0,0,0
    
    # Calculate the critical set
    iterations = 0
    while(True):
        iterations += 1
        cs = get_cs(model, x)
        
        # Invert sort the critical set by the distance to the nearest neighbor
        cs = [c for _,c in sorted(zip(nearest_neighbors,cs))]
        cs = reversed(cs)
        
        # Manipulate and undo manipulations that increase the confidence 
        # of the network
        for i in cs:
            _replace = x[i]
            x[i] = [0,0,0] 
            conf, cl = predict(x,model)
            if(monotone and iterations > 1800):
                montone = False
            if(monotone and conf <= conf_i):
                conf_i = conf
            elif(monotone):
                x[i] = _replace
                continue
            sys.stdout.write("Points occluded: %s conf: %s \r"%(points_occluded, conf))
            sys.stdout.flush()
            if(cl != y):
                # lets refine the adversarial example:
                actually_removed = []
                actually_removed_ind = []
                for i in range(len(removed)):
                    x[removed_ind[i]] = removed[i]
                    conf, cl = predict(x,model)
                    if(cl == y):
                        x[removed_ind[i]] = [0,0,0]
                        actually_removed.append(removed[i])
                        actually_removed_ind.append(removed_ind[i])
                        points_occluded -= 1
                    sys.stdout.write("Points occluded: %s conf: %s \r"%(points_occluded, conf))
                    sys.stdout.flush()
                #print " "
                return len(actually_removed), x, actually_removed, actually_removed_ind                 
            # Without refinement
            #if(cl != y):
            #    return len(removed), x, removed, removed_ind 
            #if(conf >= confidences[-1] and iterations == 0):
            #    x[i] = _replace
            #    continue
            removed.append(_replace)
            removed_ind.append(i)
            points_occluded += 1
            confidences.append(conf)
        conf, cl = predict(x,model)
        if(points_occluded > 1024 and montone == False):
            print("STARTING OVER")
            ISO_point(inp, label, model, monotone=True)
        elif(points_occluded > 1024):
            break
    print "Misclassification via Occlusion is impossible"
    
    
