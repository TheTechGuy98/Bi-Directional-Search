import sys
from pacman import runGames, readCommand
from scipy.stats import sem
from scipy.stats import t
import math

def ttest(data1,data2):
	mean1, mean2 = sum(data1)/len(data1), sum(data2)/len(data2)
	se1, se2 = sem(data1), sem(data2)
	sed = math.sqrt(se1**2.0 + se2**2.0)
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value
	cv = t.ppf(1.0 - 0.05, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p



if __name__ == '__main__':
    """
    The main function called when pacman.py is run
    from the command line:

    > python pacman.py

    See the usage string for more details.

    > python pacman.py --help
    """
    fn_list = sys.argv[1:]
    maze_list = ['mediumMaze','tinyMaze']


    res = []

    
    for i in range(len(fn_list)):
        res.append([])
        for j in maze_list:
            arg_list = ['-l',j,'-p','SearchAgent','-a','fn='+fn_list[i],'-q']
            args = readCommand( arg_list) 
            games = runGames( **args )
            scores = [game.state.getScore() for game in games]
            res[i].append(sum(scores) / float(len(scores)))

    print(ttest(res[0],res[1]))



    # import cProfile
    # cProfile.run("runGames( **args )")
    pass
