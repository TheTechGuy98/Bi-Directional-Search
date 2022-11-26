# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import manhattanDistance
from collections import defaultdict

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state
        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    seen = []
    stack = util.Stack()  # Stack Fringe for DFS
    stack.push((problem.getStartState(), []))
    while stack.isEmpty() == False:
        element = stack.pop()
        node = element[0]
        pathTillNode = element[1]

        if problem.isGoalState(node) == True:  # Solution Found
            break
        else:
            if node not in seen:  # If node was not already explored add to explored list
                seen.append(node)
                listOfChildren = problem.getSuccessors(node)
                for child in listOfChildren:
                    stack.push((child[0], pathTillNode + [child[1]]))  # Push Child Nodes in Stack
    return pathTillNode


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    seen = []
    queue = util.Queue()  # Queue Fringe for BFS
    queue.push((problem.getStartState(), []))
    while queue.isEmpty() == False:
        element = queue.pop()
        node = element[0]
        pathTillNode = element[1]
        if problem.isGoalState(node) == True:  # Solution Found
            break
        else:
            if node not in seen:  # If node was not already explored add to explored list
                seen.append(node)
                listOfChildren = problem.getSuccessors(node)
                for child in listOfChildren:
                    queue.push((child[0], pathTillNode + [child[1]]))  # Push Child Nodes in queue
    return pathTillNode


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    seen = []
    priorityQueue = util.PriorityQueue()  # UCS uses Priority Queue for Fringe
    priorityQueue.push((problem.getStartState(), [], 0), 0)
    while priorityQueue.isEmpty() == False:
        element = priorityQueue.pop()
        node = element[0]
        pathTillNode = element[1]
        costTillNode = element[2]
        if problem.isGoalState(node) == True:  # Solution Found
            break
        else:
            if node not in seen:  # If node was not already explored add to explored list
                seen.append(node)
                listOfChildren = problem.getSuccessors(node)
                for child in listOfChildren:
                    priorityQueue.push((child[0], pathTillNode + [child[1]], costTillNode + child[2]),
                                       costTillNode + child[2])  # Push Child Nodes in priority queue
    return pathTillNode


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    seen = []
    priorityQueue = util.PriorityQueue()  # Astar also uses Priority Queue for Fringe
    priorityQueue.push((problem.getStartState(), [], 0),
                       heuristic(problem.getStartState(), problem) + 0)  # Adding heuristic function to the cost.
    while priorityQueue.isEmpty() == False:
        element = priorityQueue.pop()
        node = element[0]
        pathTillNode = element[1]
        costTillNode = element[2]
        if problem.isGoalState(node) == True:  # Solution Found
            break
        else:
            if node not in seen:  # If node was not already explored add to explored list
                seen.append(node)
                listOfChildren = problem.getSuccessors(node)
                for child in listOfChildren:
                    priorityQueue.push((child[0], pathTillNode + [child[1]], costTillNode + child[2]),
                                       costTillNode + child[2] + heuristic(child[0],
                                                                           problem))  # Push Child Nodes in priority queue
    return pathTillNode


def fminf(openf):
    #heap=p,index,[state,action,g,h,f]
    temp = float('inf')
    for i in range(len(openf.heap)):
        if temp > openf.heap[i][2][4]:
            temp = openf.heap[i][2][4]
    return temp


def fminb(openb):
    temp = float('inf')
    for i in range(len(openb.heap)):
        if temp > openb.heap[i][2][4]:
            temp = openb.heap[i][2][4]
    return temp


def gminf(openf,closedf):
    temp1 = float('inf')
    for i in range(len(openf.heap)):
        if temp1 > openf.heap[i][2][2]:
            temp1 = openf.heap[i][2][2]
    
    
    temp2 = float('inf')
    for i in range(len(closedf.heap)):
        if temp2 > closedf.heap[i][2][2]:
            temp2 = closedf.heap[i][2][2]
    return min(temp1,temp2)


def gminb(openb,closedb):
    temp1 = float('inf')
    for i in range(len(openb.heap)):
        if temp1 > openb.heap[i][2][2]:
            temp1 = openb.heap[i][2][2]
    
    temp2 = float('inf')
    for i in range(len(closedb.heap)):
        if temp2 > closedb.heap[i][2][2]:
            temp2 = closedb.heap[i][2][2]
    return min(temp1,temp2)


def check(openf, closedf, state, gfc):
    flag_f = False
    flag_b = False
    openf_index = 0
    closedf_index = 0
    check = False
    gfc_check = False
    
    #print("im in check openf",openf.heap)
    #print("im in check closedf",closedf.heap)
    
    for i in range(len(openf.heap)):
        if state[0] == openf.heap[i][2][0]:
            flag_f = True
            openf_index = i
            break

    for i in range(len(closedf.heap)):
        if state[0] == closedf.heap[i][2][0]:
            flag_b = True
            closedf_index = i
            break

    if flag_f == True: 
        if openf.heap[openf_index][2][2] <= gfc:
            gfc_check = True
            '''
            print('\n\n\n')
            print("index is:",openf_index)
            print("state in openf",state)
            print("The old cost:",openf.heap[openf_index][2][2])
            print("The new cost:",gfc)
            '''
        else:
            gfc_check = False

    if flag_b==True:
        if closedf.heap[closedf_index][2][2] <= gfc:
            gfc_check = True
            '''
            print('\n\n\n')
            print("index is:",closedf_index)
            print("state in closedf",state)
            print("The old cost:",closedf.heap[closedf_index][2][2])
            print("The new cost:",gfc)
            '''
        else:
            gfc_check = False

    return openf_index, closedf_index, flag_f, flag_b, gfc_check


def check_in_openb(state, openb):
    for i in range(len(openb.heap)):
        if state[0] == openb.heap[i][2][0]:
            return True, openb.heap[i][2][2]
    return False, 0


def check_in_openf(state, openf):
    for i in range(len(openf.heap)):
        if state[0] == openf.heap[i][2][0]:
            return True, openf.heap[i][2][2]
    return False, 0

def reverse_actions_mm0(final_actions_openb):
    reversed_actions=[]
    final_actions_openb=final_actions_openb[:-1]
    for action in final_actions_openb:
        if action == 'West':
            reversed_actions.append('East')
        elif action == 'East':
            reversed_actions.append('West')
        elif action == 'South':
            reversed_actions.append('North')
        elif action == 'North':
            reversed_actions.append('South')
    return reversed_actions[::-1]



def reverse_actions(final_actions_openb):
    final_actions_openb.reverse()
    for i in range(len(final_actions_openb)):
        if final_actions_openb[i] == 'North':
            final_actions_openb[i]='South'
        elif final_actions_openb[i] == 'West':
            final_actions_openb[i]='East'
        elif final_actions_openb[i] == 'East':
            final_actions_openb[i]='West'
        else: 
            final_actions_openb[i]='North'
    return final_actions_openb



def biderictional_MM0(problem):
    # Set up fringe and closed set for both forward and backward direction
    open_f = util.Queue()
    closed_f = defaultdict()
    open_b = util.Queue()
    closed_b = defaultdict()

    # for forward direction push start state and no action
    open_f.push((problem.getStartState(), []))
    # for backward direction push goal state and no action
    open_b.push((problem.goal, []))

    # keep running while either fringe is not empty
    while not open_f.isEmpty() and not open_b.isEmpty():
        # pop out the next node and actions to reach that node in forward direction
        curr_node_f, action_f = open_f.pop()

        # Check if the current node has been reached in reverse direction, this will mean we have found path
        if (curr_node_f in closed_b):
            # return forward action and reversed backward actions
            return action_f + reverse_actions_mm0(closed_b[curr_node_f])

        # check if current node has been explored previously
        if curr_node_f not in closed_f:
            # Calling for all successors of current node
            curr_successors = problem.getSuccessors(curr_node_f)

            # for each successor node if it is not already explored
            for succ_node, action, _ in curr_successors:
                if succ_node not in closed_f:
                    # push to forward fringe with action to reach this node
                    open_f.push((succ_node, action_f + [action]))
                    closed_f[curr_node_f] = action_f + [action]

        ## Now repeat same for backward search ##

        # pop out the next node and actions to reach that node in reverse direction
        curr_node_b, action_b = open_b.pop()

        # Check if the current node has been reached in forward direction, this will mean we have found path
        if curr_node_b in closed_f:
            problem.display_expanded_nodes()
            return closed_f[curr_node_b] + reverse_actions_mm0(action_b)

        # check if current node has been explored previously
        if curr_node_b not in closed_b:
            # Calling for all successors of current node
            curr_successors = problem.getSuccessors(curr_node_b)

            # for each successor node if it is not already explored
            for succ_node, action, _ in curr_successors:
                if succ_node not in closed_b:
                    # push to backward fringe with action to reach this node
                    open_b.push((succ_node, action_b + [action]))
                    closed_b[curr_node_b] = action_b + [action]
    return []


def bidirectionalsearch(problem, heuristic=nullHeuristic):
    "*** YOUR CODE HERE ***"
    '''
    ||||||||||||Referenceblock||||||||||||||
    problem.getSuccessors(next_state)
    problem.getCostOfActions(new_path)
    print(problem.getStartState()) #(5, 5)
    print(problem.getSuccessors(problem.getStartState())) #[((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    states=problem.getSuccessors(problem.getStartState()) 
    print(states[0][0])#(5, 4)
    util.raiseNotDefined()
    heap stores in format p,priority order number (or index basically) and then [state,g,h,f]
    heap=p,index,[state,action,g,h,f]
    print(openf.heap[0][2])
    print(openf.heap[0][0])
    '''

    # For simplification interms of understanding the algorithm we are avoiding the use of getCostOfActions and plus since we store the g value it makes things a bit more faster
    openf = util.PriorityQueue()
    openb = util.PriorityQueue()
    closedf = util.PriorityQueue()
    closedb = util.PriorityQueue()
    
    #Creating the copy of problem and modifying it so it can be used in theheuristic function as argument
    
    initialState = problem.getStartState()
    goalState = problem.goal
    
    import copy
    problemF = copy.deepcopy(problem)
    problemB = copy.deepcopy(problem)
    problemB.startState = goalState
    problemB.goal = initialState

    # We will use [state,g,h,f],p format where p=max(f,2*g) and f=g+h
    openf.push([problemF.getStartState(), [], 0, 0, heuristic(problemF.getStartState(),problemF)], heuristic(problemF.getStartState(),problemF))
    openb.push([problemB.getStartState(), [], 0, 0, heuristic(problemB.getStartState(),problemB)], heuristic(problemB.getStartState(),problemB))

    #print(openf.heap)
    
    # Variables and other declarations
    U = float('inf')
    final_actions_openf = []
    final_actions_openb = []
    
    
    '''
    
    initialState = problem.getStartState()
    goalState = problem.goal    
    problemF = problem
    problemB = problemF
    problemB.startState = goalState
    problemB.goal = initialState
    '''
    
    #print("Start state:",problem.getStartState())
    #print("Goal state:",problem.goal)
    
    count=0
    while not (openf.isEmpty() and openb.isEmpty()):
        count+=1
        '''
        count+=1
        print("\n\n\n")
        print(count) 
        print("openf:",len(openf.heap))
        print("closedf:",len(closedf.heap))
        print("openb:",len(openb.heap))
        print("closedb:",len(closedb.heap))
        temp=[]
        for i in range(len(openf.heap)):
            temp.append(openf.heap[i][2][0])
        print("openf heap:",temp)   
        temp=[]
        for i in range(len(closedf.heap)):
            temp.append(closedf.heap[i][2][0]) 
        print("closedf heap:",temp) 
        temp=[]
        for i in range(len(openb.heap)):
            temp.append(openb.heap[i][2][0])
        print("openb heap:",temp) 
        temp=[]
        for i in range(len(closedb.heap)):
            temp.append(closedb.heap[i][2][0])
        print("closedb heap:",temp) 
        '''
        
        
        
        # Get C value
        #prmin = [openf.heap[0][2][0], openb.heap[0][2][0]]
        prmins_values = [openf.heap[0][0], openb.heap[0][0]]
        
        #print("prmin values",prmins_values)
        
        C = min(prmins_values)
        
        #print("C value:",C)
        

        # Testing whether the cost is below threshold
        f_minf = fminf(openf)  # min f value over all the states in openf
        f_minb = fminb(openb)  # min f value over all the states in openb
        g_minf = gminf(openf,closedf)  # min g value over all the states in openf
        g_minb = gminb(openb,closedb)  # min g value over all the states in openb
        #print("C, fminf , fmib,gminf,gminb: ",C, f_minf, f_minb, g_minf, g_minb)
            
        # 1 here is the epsilon value representing the min edge value that will exist in the graph or the search problem
        if U <= max(C, f_minf, f_minb, g_minf + g_minb+1): #+1
            #print("C, fminf , fmib,gminf+gmin+1: ",C, f_minf, f_minb, g_minf + g_minb+ 1)
            break

        # Now for the main part we will go through the next states
        if prmins_values.index(min(prmins_values)) == 0:  # which we are in openf condition
            
            #heap=p,index,[state,action,g,h,f]
            index=0
            temp_value=float('inf')
            for i in range(len(openf.heap)):
                if openf.heap[i][0]==C:
                    if temp_value>openf.heap[i][2][2]:
                        temp_value=openf.heap[i][2][2]
                        priority = openf.heap[i][0]
                        index=i
                    
            # removal of the parent state from openf and transfer to closedf
            #print("Parent chosen:",openf.heap[index][2][0])
            
            states = problem.getSuccessors(openf.heap[index][2][0])
            #print("The successors are:",states)
            #print("\n\n\n")
            gfn = openf.heap[index][2][2]  
            #print("Parent cost:",gfn)
            previous_action = openf.heap[index][2][1]            
            closedf.push(openf.heap[index][2],priority)            
            del openf.heap[index]
            
            
            
            for state in states:                
                # We calculate the g,f values for the child node
                action = state[1]
                gfc = gfn + state[2]
                #print("gfc value: ", gfc)
                #print("gfn value: ", gfn)
                if heuristic == nullHeuristic:
                    h = 0
                else:
                    h = heuristic(state[0], problemF)
                f = gfc + h
                
                
                
                # This function should check if the node aleady exists in the Queues or not and if it is and the g value is small then update otherwise skip this child and move to the next one
                openf_index, closedf_index, flag_f, flag_b, gfc_check = check(openf, closedf, state, gfc)
                if (flag_f or flag_b) and gfc_check:
                    #print("gfc check satisified node:",state[0])
                    #print("\n\n\n")
                    continue
                if flag_f:
                    del openf.heap[openf_index]
                if flag_b:
                    del closedf.heap[closedf_index]                  
                             
                
                # add this new state to the relevant queue
                #print("state:",state[0])
                #print("F values:",f)
                #print("2 * gfc:", 2*gfc)
                openf.push([state[0], previous_action + [action], gfc, h, f], max(f, 2 * gfc))
                
                
                # if this new state is in the other open queue then we have found the common node and now its time to return the actions
                flag, gbc = check_in_openb(state, openb)
                if flag:
                    U = min(U, gfc+gbc)
                    #print(U)
                    final_actions_openf = previous_action + [action]

                    # heap=p,index,[state,action,g,h,f] <-- just a reference for me to understand the indices
                    for i in range(len(openb.heap)):
                        if state[0] == openb.heap[i][2][0]:
                            final_actions_openb = openb.heap[i][2][1]
            
            #openf.heap=openf.heap[::-1]
            
        else:
            
            
            #heap=p,index,[state,action,g,h,f]
            index=0
            temp_value=float('inf')
            temp_state=openb.heap[0]
            for i in range(len(openb.heap)):
                if openb.heap[i][0]==C:
                    if temp_value>openb.heap[i][2][2]:
                        temp_value=openb.heap[i][2][2]
                        priority = openb.heap[i][0]
                        index=i
                    
            # removal of the parent state from openb and transfer to closedb
            #print("Parent chosen:",openb.heap[index][2][0])
            states = problem.getSuccessors(openb.heap[index][2][0]) 
            #print("The successors are:",states)
            #print("\n\n\n")
            gbn = openb.heap[index][2][2]  
            previous_action = openb.heap[index][2][1]
            closedb.push(openb.heap[index][2],priority)            
            del openb.heap[index]
            
            
            for state in states:
                
                action = state[1]
                gbc = gbn + state[2]
                if heuristic == nullHeuristic:
                    h = 0
                else:
                    h = heuristic(state[0], problemB)
                f = gbc + h

                openb_index, closedb_index,flag_f ,flag_b, gfc_check = check(openb, closedb, state, gbc)
                
                                    
                if (flag_f or flag_b) and gfc_check:
                    #print("GFC check node:",state[0])
                    #print("\n\n\n")
                    continue
                if flag_f:
                    del openb.heap[openb_index]
                if flag_b:
                    del closedb.heap[closedb_index] 
        

                # add this new state
                #print("state:",state[0])
                #print("F values:",f)
                #print("2 * gbc:", gbc)
                openb.push([state[0], previous_action + [action], gbc, h, f], max(f, 2 * gbc))
                
                flag, gfc = check_in_openf(state, openf)
                if flag:
                    U = min(U, gfc+gbc)
                    #print(U)
                    final_actions_openb = previous_action + [action]

                    # heap=p,index,[state,action,g,h,f]
                    for i in range(len(openf.heap)):
                        if state[0] == openf.heap[i][2][0]:
                            final_actions_openf = openf.heap[i][2][1]
            
            #openb.heap=openb.heap[::-1]
        '''
        print("\nAt the end of the cycle!")
        print("openf:",len(openf.heap))
        print("closedf:",len(closedf.heap))
        print("openb:",len(openb.heap))
        print("closedb:",len(closedb.heap))
        temp=[]
        for i in range(len(openf.heap)):
            temp.append(openf.heap[i][2][0])
        print("openf heap:",temp)   
        temp=[]
        for i in range(len(closedf.heap)):
            temp.append(closedf.heap[i][2][0]) 
        print("closedf heap:",temp) 
        temp=[]
        for i in range(len(openb.heap)):
            temp.append(openb.heap[i][2][0])
        print("openb heap:",temp) 
        temp=[]
        for i in range(len(closedb.heap)):
            temp.append(closedb.heap[i][2][0])
        print("closedb heap:",temp) 
        '''  
        
    print("Action policy calculated!")
    print(U)
    
    '''
    print(openf.heap[0][2])
    print(openf.heap[0][2][0])
    print(openf.heap[0][2][1])
    print(openf.heap[0][2][2])
    print(openf.heap[0][2][3])
    print(openf.heap[0][2][4])
    '''
    #print(final_actions_openf + reverse_actions(final_actions_openb))
    problem.display_expanded_nodes()
    return final_actions_openf + reverse_actions(final_actions_openb)


def bidirectionalSearch(problem, heuristic=nullHeuristic, useEpsilon=False, useFractional=False, pv=0.5):
    if useEpsilon:
        e = 1
    else:
        e = 0
    initialState = problem.getStartState()
    goalState = problem.goal

    import copy
    problemF = copy.deepcopy(problem)
    problemB = copy.deepcopy(problem)
    problemB.startState = goalState
    problemB.goal = initialState

    gF, gB = {initialState: 0}, {goalState: 0}
    openF, openB = [initialState], [goalState]
    closedF, closedB = [], []
    parentF, parentB = {}, {}
    U = float('inf')

    if useFractional:
        p = pv
    else:
        p = 0.5

    def extend(U, open_dir, open_other, g_dir, g_other, closed_dir, parent, search_direction):
        """Extend search in given direction"""
        n = find_key(C, open_dir, g_dir, search_direction)
        #print("parent chosen:",n)
        open_dir.remove(n)
        closed_dir.append(n)

        for (c, direction, cost) in problem.getSuccessors(n):
            if c in open_dir or c in closed_dir:
                if g_dir[c] <= g_dir[n] + cost:
                    continue
                
                if c in open_dir:
                    open_dir.remove(c)
                else:
                    closed_dir.remove(c)

            g_dir[c] = g_dir[n] + cost
            #print("gfc value: ",g_dir[c])
            #print("gfn value: ",g_dir[n])
            if search_direction == 'F':
                f = g_dir[c] + heuristic(c, problemF)
                #print("state:",c)
                #print("F values:",f)
                #print("2 * gfc:", 2*g_dir[c])
            else:
                f = g_dir[c] + heuristic(c, problemB)
                #print("state:",c)
                #print("F values:",f)
                #print("2 * gbc:", 2*g_dir[c])
                
            open_dir.append(c)
            parent[c] = (n, direction)

            if c in open_other:
                U = min(U, g_dir[c] + g_other[c])

        return U, open_dir, closed_dir, g_dir

    def find_min(open_dir, g, search_direction):
        """Finds minimum priority, g and f values in open_dir"""
        # pr_min_f isn't forward pr_min instead it's the f-value
        # of node with priority pr_min.
        pr_min, pr_min_f = float('inf'), float('inf')
        temp=[]
        
        for n in open_dir:
            if search_direction == 'F':
                f = g[n] + heuristic(n, problemF)
            else:
                f = g[n] + heuristic(n, problemB)

            if useEpsilon:
                pr = max(f, 2 * g[n] + 1)
            elif useFractional:
                if search_direction == 'F':
                    pr = max(f, g[n] / float(p))
                else:
                    minus_p = 1-float(p)    
                    pr = max(f, g[n] / minus_p)
            else:
                pr = max(f, 2 * g[n])
            pr_min = min(pr_min, pr)
            #print("prmin:",pr_min)
            pr_min_f = min(pr_min_f, f)
            #print("prmin_f:",pr_min_f)
            temp.append(g[n])
            
        
        return pr_min, pr_min_f, min(g.values())

    def find_key(pr_min, open_dir, g, search_direction):
        """Finds key in open_dir with value equal to pr_min
        and minimum g value."""
        m = float('inf')
        node = None
        for n in open_dir:
            if search_direction == 'F':
                if useEpsilon:
                    pr = max(g[n] + heuristic(n, problemF), 2 * g[n] + 1)
                elif useFractional:
                    pr = max(g[n] + heuristic(n, problemF), g[n] / float(p))
                else:
                    pr = max(g[n] + heuristic(n, problemF), 2 * g[n])
            else:
                if useEpsilon:
                    pr = max(g[n] + heuristic(n, problemB), 2 * g[n] + 1)
                elif useFractional:
                    minus_p = 1-float(p)    
                    pr = max(g[n] + heuristic(n, problemB), g[n] / minus_p)
                else:
                    pr = max(g[n] + heuristic(n, problemB), 2 * g[n])
            if pr == pr_min:
                if g[n] < m:
                    m = g[n]
                    node = n

        return node

    count=0
    while openF and openB:
        count+=1
        '''
        print("\n\n\n")
        print(count)        
        print("openf:",len(openF))
        print("closedf:",len(closedF))
        print("openb:",len(openB))
        print("closedb:",len(closedB))
        
        print("openf heap:",openF)   
        
        print("closedf heap:",closedF) 
        
        print("openb heap:",openB) 
        
        print("closedb heap:",closedB) 
        print("\n\n\n")
        '''
        
        pr_min_f, f_min_f, g_min_f = find_min(openF, gF, 'F')
        pr_min_b, f_min_b, g_min_b = find_min(openB, gB, 'B')
        C = min(pr_min_f, pr_min_b)
        #print("prmin_values:",pr_min_f, pr_min_b)

        if U <= max(C, f_min_f, f_min_b, g_min_f + g_min_b + 1):
            #print("C,fmif,fminb,gminf,gminb+e :",C, f_min_f, f_min_b, g_min_f + g_min_b + e)
            #print(e)
            # Get an interesect node between openF and openB
            intersect_list = list(set(openF) & set(openB))
            # intersect exists
            if intersect_list:
                intersect = intersect_list[0]
            # intersect does NOT exist
            else:
                if len(openF) <= len(openB):
                    intersect = openF[0]
                    openB.append(intersect)
                else:
                    intersect = openB[0]
                    openF.append(intersect)
            actionsF = []
            # Get actions from intersect to initState
            state = intersect
            while state in parentF.keys():
                (state, direction) = parentF[state]
                actionsF.append(direction)
                if state == initialState:
                    break
            actionsF = actionsF[::-1]
            # Get actions from intersect to goalState
            actionsB = []
            state = intersect
            while state in parentB.keys():
                (state, direction) = parentB[state]
                actionsB.append(direction)
                if state == goalState:
                    break
            # solution
            solution = actionsF
            for action in actionsB:
                if action == 'North':
                    solution.append('South')
                if action == 'East':
                    solution.append('West')
                if action == 'South':
                    solution.append('North')
                if action == 'West':
                    solution.append('East')
            # return U
            #print(solution)
            return solution

        if C == pr_min_f:
            # Extend forward
            #print("C value",C)
            U, openF, closedF, gF = extend(U, openF, openB, gF, gB, closedF, parentF, 'F')
            
        else:
            # Extend backward
            #print("C value",C)
            U, openB, closedB, gB = extend(U, openB, openF, gB, gF, closedB, parentB, 'B')
        #print("C,fmif,fminb,gminf,gminb+e :",C, f_min_f, f_min_b, g_min_f,g_min_b)
        #print(U)
    return float('inf')



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

# Final project bidirectional search
bid = bidirectionalsearch
bid_mm0 = biderictional_MM0
bis=bidirectionalSearch
