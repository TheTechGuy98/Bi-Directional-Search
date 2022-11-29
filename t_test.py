import sys
from pacman import runGames, readCommand
from scipy.stats import ttest_ind
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
    maze_list = ['bigMaze','openMaze','mediumMaze','customBigMaze1','customBigMaze2','customMediumMaze1','customMediumMaze2','customMediumMaze3']
    fn_list = fn_list_string[1:-1].split(':')
    deleteFile()



    
    for i in range(len(fn_list)):
        for j in maze_list:
            f = open(".\data\data.txt", "a")
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
