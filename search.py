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
    return  [s, s, w, s, w, w, s, w]

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
    stack = util.Stack()     # Stack Fringe for DFS
    stack.push((problem.getStartState(), []))
    while stack.isEmpty() == False:
        element = stack.pop()
        node = element[0]
        pathTillNode = element[1]

        if problem.isGoalState(node) == True:       # Solution Found    
            break
        else:
            if node not in seen:            # If node was not already explored add to explored list
                seen.append(node)
                listOfChildren = problem.getSuccessors(node)
                for child in listOfChildren:
                    stack.push((child[0], pathTillNode+[child[1]]))    # Push Child Nodes in Stack
    return pathTillNode

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    seen = [] 
    queue = util.Queue()     # Queue Fringe for BFS
    queue.push((problem.getStartState(), []))
    while queue.isEmpty() == False:
        element = queue.pop()
        node = element[0]
        pathTillNode = element[1]
        if problem.isGoalState(node) == True:       # Solution Found    
            break
        else:
            if node not in seen:            # If node was not already explored add to explored list 
                seen.append(node)
                listOfChildren = problem.getSuccessors(node)
                for child in listOfChildren:
                    queue.push((child[0], pathTillNode+[child[1]]))    # Push Child Nodes in queue
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
        if problem.isGoalState(node) == True:       # Solution Found 
            break
        else:
            if node not in seen:     # If node was not already explored add to explored list 
                seen.append(node)  
                listOfChildren = problem.getSuccessors(node)
                for child in listOfChildren:
                    priorityQueue.push((child[0], pathTillNode + [child[1]], costTillNode+child[2]),costTillNode+child[2])  # Push Child Nodes in priority queue
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
    priorityQueue.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem) + 0)  #Adding heuristic function to the cost.
    while priorityQueue.isEmpty() == False:
        element = priorityQueue.pop()
        node = element[0]
        pathTillNode = element[1]
        costTillNode = element[2]
        if problem.isGoalState(node) == True:       # Solution Found 
            break
        else:
            if node not in seen:     # If node was not already explored add to explored list 
                seen.append(node)  
                listOfChildren = problem.getSuccessors(node)
                for child in listOfChildren:
                    priorityQueue.push((child[0], pathTillNode + [child[1]], costTillNode + child[2]),costTillNode + child[2] + heuristic(child[0], problem))  # Push Child Nodes in priority queue
    return pathTillNode


def fminf(openf):
    temp=float('inf')
    for i in range(len(openf.heap)):
        if temp>openf.heap[i][2][4]:
            temp=openf.heap[i][2][4]
    return temp

def fminb(openb):
    temp=float('inf')
    for i in range(len(openb.heap)):
        if temp>openb.heap[i][2][4]:
            temp=openb.heap[i][2][4]
    return temp

def gminf(openf):
    temp=float('inf')
    for i in range(len(openf.heap)):
        if temp>openf.heap[i][2][2]:
            temp=openf.heap[i][2][2]            
    return temp

def gminb(openb):
    temp=float('inf')
    for i in range(len(openb.heap)):
        if temp>openb.heap[i][2][2]:
            temp=openb.heap[i][2][2]
    return temp


def check(openf,closedf,state,gfc):  
    flag_f=False
    flag_b=False
    openf_index=0
    closedf_index=0
    
    for i in range(len(openf.heap)):
        if state[0]==openf.heap[i][2][0]:
            flag_f=True
            openf_index=i
    
    for i in range(len(closedf.heap)):
        if state[0]==closedf.heap[i][2][0]:
            flag_b=True
            closedf_index=i
            
    if flag_f==True and flag_b==True:
        if openf.heap[openf_index][2][2]<=gfc:
            return True
        else:
            return False
        
def check_in_openb(state,openb):
    for i in range(len(openb.heap)):
        if state[0]==openb.heap[i][2][0]:
            return True, openb.heap[i][2][2]
    return False,0
    
    
def check_in_openf(state,openf):
    for i in range(len(openf.heap)):
        if state[0]==openf.heap[i][2][0]:
            return True, openf.heap[i][2][2]
    return False,0   

def reverse_actions(final_actions_openb):
    
    print(final_actions_openb.reverse())
    #final_actions_openb=final_actions_openb.reverse()
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
    
    

def bidirectionalsearch(problem,heuristic=nullHeuristic):
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
    
    #For simplification interms of understanding the algorithm we are avoiding the use of getCostOfActions and plus since we store the g value it makes things a bit more faster
    openf=util.PriorityQueue()
    openb=util.PriorityQueue()
    closedf=util.PriorityQueue()
    closedb=util.PriorityQueue()
    
    #We will use [state,g,h,f],p format where p=max(f,2*g) and f=g+h
    openf.push([problem.getStartState(),[],0,0,0],0) 
    openb.push([problem.goal,[],0,0,0],0) 
    
    #Variables and other declarations
    U=float('inf')
    final_actions_openf=[]
    final_actions_openb=[]

    
    
    while not (openf.isEmpty() and openb.isEmpty()):
        
        
        '''
        print("\n\n\n New epoch \n\n\n")
        print("openf: ",openf.heap)
        print("openb: ",openb.heap)
        print("closedb: ",closedb.heap)
        print("closedf: ",closedf.heap)
        '''
        
        #Get C value
        prmin=[openf.heap[0][2][0],openb.heap[0][2][0]]
        prmins_values=[openf.heap[0][0],openb.heap[0][0]]
        C=min(prmins_values)        
        #print("C_value:",C)       
        
        #Testing whether the cost is below threshold
        f_minf=fminf(openf) #min f value over all the states in openf
        f_minb=fminb(openb) #min f value over all the states in openb
        g_minf=gminf(openf) #min g value over all the states in openf
        g_minb=gminb(openb) #min g value over all the states in openb
        
        #1 here is the epsilon value representing the min edge value that will exist in the graph or the search problem          
        if U<= max(C,f_minf,f_minb,g_minf+g_minb+1): 
            '''
            print("fminf value:",f_minf)            
            print("fminb value:",f_minb)            
            print("gminf value:",g_minf)            
            print("gminb value:",g_minb)
            print("Exit!!")  
            '''              
            break
            
          
        #Now for the main part we will go through the next states        
        if prmins_values.index(min(prmins_values))==0: #which we are in openf condition
            #print("Choosing state",prmin[0]," for expansion and this is openf condition")
            
            states=problem.getSuccessors(prmin[0]) #prmin[0] is prminf in pseudo code
            
            #removal of the parent state from openf and closedf
            gfn=openf.heap[0][2][2] #same as prmin[0][2] (i think if the indexing is done that way)
            priority=openf.heap[0][0]
            previous_action=openf.heap[0][2][1]
            closedf.push(openf.pop(),priority)
            
            
            for state in states:
                
                '''
                    ||||||||||||Referenceblock||||||||||||||
                    print(problem.getSuccessors(problem.getStartState())) #[((5, 4), 'South', 1), ((4, 5), 'West', 1)]
                    states=problem.getSuccessors(problem.getStartState()) 
                    print(states[0][0])#(5, 4)
                    check if state is in openf and closedf and if the value is less then update
                    heap=p,index,[state,action,g,h,f] adding reference here
                '''
                
                #We calculate the g,f values for the child node
                action=state[1]
                gfc=gfn+state[2]
                if heuristic==nullHeuristic:
                    h=0
                else:
                    h=manhattanDistance(state[0],problem.goal)
                f=gfc+h
                
                #This function should check if the node aleady exists in the Queues or not and if it is and the g value is small then update otherwise skip this child and move to the next one
                if check(openf,closedf,state,gfc):
                    #print("The check is True")
                    continue
                else: 
                    #add this new state to the relevant queue
                    openf.push([state[0],previous_action+[action],gfc,h,f],max(f,2*gfc))
                    
                    #if this new state is in the other open queue then we have found the common node and now its time to return the actions
                    flag,gbc=check_in_openb(state,openb)
                    if flag:                        
                        U=min(U,gfc,gbc)
                        print("The new U value now: ",U)
                        print("The common state is: ",state[0])
                        final_actions_openf=previous_action+[action]
                        
                        #heap=p,index,[state,action,g,h,f] <-- just a reference for me to understand the indices
                        for i in range(len(openb.heap)):
                            if state[0] == openb.heap[i][2][0]:
                                final_actions_openb=openb.heap[i][2][1]
             
            '''    
            print("openf: ",openf.heap)
            print("openb: ",openb.heap)
            print("closedb: ",closedb.heap)
            print("closedf: ",closedf.heap)
            '''
            
        else:
            #Do exactly the same here as above except we will reverse/change the queues and everything else        
            states=problem.getSuccessors(prmin[1]) #prmin[1] is prminb in pseudo code
            
            #print("Choosing state",prmin[1]," for expansion and this is openb condition")
            
            #removal of the parent state from openf and closedf
            gbn=openb.heap[0][2][2] #same as prmin[0][2] (i think if the indexing is done that way)
            priority=openb.heap[0][0]
            previous_action=openb.heap[0][2][1]
            closedb.push(openb.pop(),priority)
            
            for state in states:    
                action=state[1]
                gbc=gbn+state[2]
                if heuristic==nullHeuristic:
                    h=0
                else:
                    h=manhattanDistance(state[0],problem.getStartState())
                f=gbc+h
                
                
                if check(openb,closedb,state,gbc):
                    continue
                else:                    
                    
                    #add this new state
                    openb.push([state[0],previous_action+[action],gbc,h,f],max(f,2*gbc))
                    flag,gbc=check_in_openf(state,openf)
                    if flag:                        
                        U=min(U,gfc,gbc)
                        print("The new U value now: ",U)
                        print("The common state is: ",state[0])
                        final_actions_openb=previous_action+[action]
                        
                        #heap=p,index,[state,action,g,h,f]
                        for i in range(len(openf.heap)):
                            if state[0] == openf.heap[i][2][0]:
                                final_actions_openf=openf.heap[i][2][1]
                       
            '''            
            print("openf: ",openf.heap)
            print("openb: ",openb.heap)
            print("closedb: ",closedb.heap)
            print("closedf: ",closedf.heap)
            '''
    print("Action policy calculated!")
    #reverse_actions will reverse all the actions in openb since this search started from the goal state and filled all the actions accordingly , hence if we were
    #travelling in forward direction we would need the order and the actions to be oppoiste: For e.g North , West becomes East , South
    return final_actions_openf+reverse_actions(final_actions_openb)
    
    #print the output at instances to see the queue values or some other variable if requried. It should clear all the doubts
    

    
    
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

#Final project bidirectional search
bid= bidirectionalsearch
