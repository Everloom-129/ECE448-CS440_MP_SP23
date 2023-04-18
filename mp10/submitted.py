
import numpy as np

epsilon = 1e-3

def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. 
    P[r, c, a, r', c'] is the probability 
    that the agent will move from cell (r, c) to (r', c')
    if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    M,N = model.M, model.N
    P = np.zeros((M,N,4,M,N))
    for r in range(M):
        for c in range(N):
            if not model.T[r, c]:
                for a in range(4):
                    movements = []
                    if a == 0:  # left
                        movements = [(0, -1), (1, 0), (-1, 0)]
                    elif a == 1:  # up
                        movements = [(-1, 0), (0, -1), (0, 1)]
                    elif a == 2:  # right
                        movements = [(0, 1), (-1, 0), (1, 0)]
                    elif a == 3:  # down
                        movements = [(1, 0), (0, 1), (0, -1)]
                    # model.D[r, c, 0]: front, 
                    # model.D[r, c, 1]: left (counter-clockwise)
                    # model.D[r, c, 2]: right (clockwise)
                    for idx, (dr, dc) in enumerate(movements):
                        r_next = r + dr
                        c_next = c + dc
                        # Boundary check
                        Bound = r_next < 0 or r_next >= model.M or c_next < 0 or c_next >= model.N 
                        if Bound or model.W[r_next, c_next]:
                            r_next, c_next = r, c

                        P[r, c, a, r_next, c_next] = model.D[r, c, idx]
            else:
                P[r, c, :, :, :] = 0

    return P
         
def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    M,N = model.M, model.N
    U_next = np.zeros((M, N))

    for r in range(M):
        for c in range(N):

            best_value = -np.inf
            for a in range(4):
                value = 0
                for r_next in range(M):
                    for c_next in range(N):
                        value += P[r, c, a, r_next, c_next] * U_current[r_next, c_next]

                best_value = max(best_value, value)

            U_next[r, c] = model.R[r, c] + model.gamma * best_value

    return U_next    



def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    max_iterations = 100

    # Compute the transition matrix
    P = compute_transition_matrix(model)

    # Initialize the utility function
    M,N = model.M, model.N
    U_current = np.zeros((M, N))

    for _ in range(max_iterations):
        # Update the utility function
        U_next = update_utility(model, P, U_current)

        # Check for convergence
        if np.all(np.abs(U_next - U_current) < epsilon):
            break

        # Update U_current for the next iteration
        U_current = U_next
    print(_)
    return U_current
    

if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
