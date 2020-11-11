import numpy as np

def readMaze(filename):
    '''
    I am attaching a zip folder with both benchmark suites, which were provided to us by Dr. David Z. Pan and Dr. Minsik Cho. Be advised, though, that the instances' files are not described and are not straightforward to comprehend. Because of that, let me give you some details on how to interpret these files.
    Regarding the test files of benchmark I, the first row indicates the grid size and a lateness constraint—although the time constraint has been invariably set to 100 units of time. Then, the subsequent rows indicate the coordinates of where the blockages appear. The remaining rows show the source and goal positions of each droplet (called net). Each droplet (net) row consists of an array of numbers that represent: net id, source x, source y, 0 (as a divider), target x, target y, manhattan distance. An example figure of Test 2 (test_12_12_2.in) is attached.
    Regarding the test files of benchmark II, each test consists of a set of subproblems. Each subproblem file has mainly the same structure as those from benchmark I in addition to some particular rows to consider. Some rows may be labeled as "xet," these indicate droplets that should merge before arriving at the target cell. The xet rows include a semicolon character to split the data of both droplets; the next two numbers on the right side of the semicolon are the source x and source y coordinates of the droplet to be merged. Finally, the "WAT" labeled rows are considered disposal cells; any time a droplet targets these cells, the droplet is deemed to be dismissed from the array.
    '''

    fd=open(filename, 'r')
    blockages, nets = [], []
    row, col = list(map(int, fd.readline().strip().split()[1:3])) #read gridsize

    # read blockages
    for line in fd:
        if line.startswith('block '):
            blockages.append(list(map(int, line.strip().split()[1:])))

        elif line.startswith('net '):
            _, src_x, src_y, _, dst_x, dst_y, _ = list(map(int, line.split()[1:]))
            nets.append(((src_x, src_y), (dst_x, dst_y)))

    maze=np.zeros((row, col))
    blockage_indices=np.concatenate([__insidepoints(*b) for b in blockages], axis=0)
    maze[blockage_indices[:, 0], blockage_indices[:, 1]]=1

    return maze, nets


def __insidepoints(x1, y1, x2, y2):

    x_coords = np.arange(min(x1, x2), max(x1, x2)+1, 1)
    y_coords = np.arange(min(y1, y2), max(y1, y2)+1, 1)
    xx, yy = np.meshgrid(x_coords, y_coords, sparse=False)
    return np.concatenate([yy.reshape(-1, 1), xx.reshape(-1, 1)], axis=1)