from queue import Queue
import heapq  
goal_state = "123456780"

successor_moves = {
    "index0": ["down", "right"],
    "index1": ["down", "left", "right"],
    "index2": ["down", "left"],
    "index3": ["up", "down", "right"],
    "index4": ["up", "down", "left", "right"],
    "index5": ["up", "down", "left"],
    "index6": ["up", "right"],
    "index7": ["up", "left", "right"],
    "index8": ["up", "left"]
}

def move(state, action):
    l = list(state) # converting string into list
    zero = l.index("0") # to find position of blank tile
    if action == "up":
        swap = zero - 3 # to move upward
    elif action == "down":
        swap = zero + 3
    elif action == "left":
        swap = zero - 1   # to move to left position
    elif action == "right":
        swap = zero + 1
    l[zero], l[swap] = l[swap], l[zero]  # swap
    return "".join(l) # to convert list into string

# printing board
def display_board(state):
    print(state[0:3])  # slicing start means inclusive,end means exclusive
    print(state[3:6])
    print(state[6:9])
    print("-" * 7)

# BFS function
def bfs(start):
    print("\n--- BFS ---")
    q = Queue()
    q.put((start, []))  # store state+action(right,left,up,down)
    visited = set([start])
    nodes_expanded = 0

    while not q.empty(): # loop stops when queue is empty
        current, path = q.get() # pop from queue
        nodes_expanded += 1
        display_board(current)  # Show each state

        if current == goal_state:  # check if current state=goal sate
            print("Goal state found!")
            return path, nodes_expanded

        zero = current.index("0")
        for action in successor_moves["index" + str(zero)]:
            new_state = move(current, action)
            if new_state not in visited:
                visited.add(new_state)
                q.put((new_state, path + [action]))

    print("Goal state not found") # if goal not found
    return None, nodes_expanded

# dfs ftn
def depth_first_search(start):
    print("\n--- DFS ---")
    stack = [(start, [])] #LIFO stack to store state and path
    visited = set([start]) #avoid same states from revisiting
    nodes_expanded = 0

    while stack:
        current, path = stack.pop() 
        nodes_expanded += 1
        display_board(current)  # displaying each state

        if current == goal_state:
            print("Goal state found!")
            return path, nodes_expanded

        zero = current.index("0") #index of '0' tile
        actions = successor_moves["index" + str(zero)]
        for action in reversed(actions): #reverse actions so that dfs goes deep in natural way
            new_state = move(current, action)
            if new_state not in visited:  #to explore not visited states 
                visited.add(new_state)
                stack.append((new_state, path + [action]))

    print("Goal state not found")
    return None, nodes_expanded

# This function find the misplaced tiles
def misplaced_tiles(state):
    count = 0
    for i in range(9):
        # If the tile is not blank (as we say 0 is blank) and is not in the goal position
        if state[i] != "0" and state[i] != goal_state[i]:
            count += 1
    return count  # Return number of misplaced tiles

# ------------------ A* ------------------
def A_star(start):
    print("\n--- A* ---")
    
    heap = [(misplaced_tiles(start), start, [])]  # priority queue (min-heap)
    visited = set()  # states we have already visited
    nodes_expanded = 0  # count of expanded nodes

    while heap:
        f, current, path = heapq.heappop(heap)  # get state with lowest f = g + h
        nodes_expanded += 1
        display_board(current)  # show each state

        if current == goal_state:  # goal check
            print("Goal state found by A*")
            return path, nodes_expanded

        if current in visited:  # skip if already visited
            continue
        visited.add(current)

        zero = current.index("0")  # find blank tile
        for action in successor_moves["index" + str(zero)]:
            new_state = move(current, action)
            if new_state not in visited:
                g = len(path) + 1 
                h = misplaced_tiles(new_state)  # heuristic
                f=g+h
                heapq.heappush(heap, (f, new_state, path + [action]))

    print("A* search did not reach the goal")
    return None, nodes_expanded

# ------------------ Main ------------------
print("Enter numbers from 0-8 where 0 is blank tile:")
R1 = input("Row 1: ").split()  # takes input in the form of list
R2 = input("Row 2: ").split()
R3 = input("Row 3: ").split()
initial_state = "".join(R1 + R2 + R3) # converting list into string

bfs_path, bfs_nodes = bfs(initial_state)  # calling bfs function
dfs_path, dfs_nodes = depth_first_search(initial_state)
astar_path, astar_nodes = A_star(initial_state)

print("\nNodes expanded:")
print("BFS:", bfs_nodes)
print("DFS:", dfs_nodes)
print("A* :", astar_nodes)

# Determine best algorithm (least nodes expanded)
best = min(bfs_nodes, dfs_nodes, astar_nodes)
if best == bfs_nodes:
    print("Best algorithm: BFS")
elif best == dfs_nodes:
    print("Best algorithm: DFS")
else:
    print("Best algorithm: A*")