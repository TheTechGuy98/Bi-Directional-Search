import sys
from pacman import runGames, readCommand
from scipy.stats import ttest_ind
from sklearn import preprocessing
import numpy as np
import math
import os

def ttest(data1,data2):
	return ttest_ind(data1,data2)


def deleteFile(path = '.\data\data.txt'):
    try:
        os.remove(path)
    finally:
        return

def readFile(path = '.\data\data.txt'):
    f = open(path,'r')
    return f.readlines()

def dataArray(lines):
    obj = {}
    for line in lines:
        lineArray = line.split(',')

        if lineArray[0] not in obj:
            obj[lineArray[0]] = [int(lineArray[1])]
        else:
            obj[lineArray[0]].append(int(lineArray[1]))
    data = []
    for fn in obj:
        data.append(obj[fn])
    return data






if __name__ == '__main__':
    """
    The main function called when pacman.py is run
    from the command line:

    > python pacman.py

    See the usage string for more details.

    > python pacman.py --help
    """
    fn_list_string = sys.argv[1]
    maze_list = ['mediumMaze','customMediumMaze2','customMediumMaze3','customMediumMaze1','customBigMaze1_t','customBigMaze2_t','customMediumMaze1_t','customMediumMaze2_t','customMediumMaze3_t','openMaze_t','customMaze2_t']
    fn_list = fn_list_string[1:-1].split(':')
    deleteFile()



    
    for i in range(len(fn_list)):
        for j in maze_list:
            f = open(".\data\data.txt", "a")
            if fn_list[i].split(',')[0] == 'bid' and fn_list[i].split(',')[1] == 'heuristic=nullHeuristic': 
                f.write('bid_mm0'+',')
            elif fn_list[i].split(',')[0] == 'bid' and fn_list[i].split(',')[1] == 'heuristic=euclideanHeuristic': 
                f.write('bid_mm_eucl'+',')
            else:
                f.write(str(fn_list[i].split(',')[0]+','))
            f.close()
            print(j)
            arg_list = ['-l',j,'-p','SearchAgent','-a','fn='+fn_list[i],'-q']
            args = readCommand( arg_list) 
            games = runGames( **args )

    lines = readFile()
    res = dataArray(lines)
    print(res)
    print(ttest(res[0],res[1]))



    # import cProfile
    # cProfile.run("runGames( **args )")
    pass
