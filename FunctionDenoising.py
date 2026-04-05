import numpy as np
import matplotlib.pyplot as plt

class TridiagonalMatrices:
    """
    Representation of a sparse tridiagonal matrix, stored using three vectors:
    a: main diagonal, length n
    b: lower diagonal, length n-1
    c: upper diagonal, length n-1
    """

    def __init__(self,
                 a: np.ndarray,
                 b: np.ndarray,
                 c: np.ndarray
                 ):
        """
        Initialise a sparse tridiagonal matrix

        Arguments:
        a: Main diagonal, NumPy array of length n
        b: Lower diagonal, NumPy array of length n-1
        c: Upper diagonal, NumPy array of length n-1
        """
        self._a = a.copy()
        self._b = b.copy()
        self._c = c.copy()

        # To ensure the LU factorization is only computed once, will be used in question 1.4
        self._A_is_decomposed = False

    @property
    def size(self) -> int:
        """
        Return the matrix dimension dimension n (number of rows/columns)
        """
        return len(self._a)
    
    @property
    def dense_rep(self) -> np.ndarray:
        """
        Return a NumPy 2D array corresponding to the dense representation of the matrix - Intended for testing and debugging only
            
        Returns:
            Dense nxn NumPy array
        """
        n = self.size
        A = np.zeros((n, n))

        # Fill diagonals with their respective arrays

        np.fill_diagonal(A, self._a)
        np.fill_diagonal(A[1:], self._b) # to choose lower diagonal, start from row 1 (and column 0)
        np.fill_diagonal(A[:, 1:], self._c) # to choose upper diagonal, start from column 1 (and row 0)
        
        return A

    def matvec_prod(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the matrix-vector product Ax using the sparse structure. Does not convert to dense format

        Arguments:
            x: Vector of length n (NumPy array)

        Returns:
            Resulting product, a vector Ax of length n
        """
        n = self.size
        product = np.zeros(n)

        # Main diagonal contribution to product: a_i * x_i for all i = 0, 1, ..., n-1
        product = self._a * x

        # Lower diagonal contribution to product: a_i * x_(i-1) for all i = 1, 2, ..., n-1
        product[1:] += self._b * x[:-1]
 
        # Upper diagonal contribution to product: a_i * x_(i+1) for all i = 0, 1, ..., n-2
        product[:-1] += self._c * x[1:]

        return product
    
    def lu_decomp(self):
        """
        Compute the LU decomposition of sparse tridiagonal matrices, and store the resulting vectors as instance variables:
            self._alpha: main diagonal of U, NumPy array of length n
            self._beta: lower diagonal of L, NumPy array of length n-1
        note the upper diagonal of U is identical to self._c, hence it need not be computed in this function 
        The iterative formula derived in notes will be the one used for the computation of our alpha and beta vectors
        """
        n = self.size

        # initialise vectors
        alpha = np.zeros(n)
        beta = np.zeros(n-1)

        alpha[0] = self._a[0]

        for i in range(1,n):
            beta[i-1] = self._b[i-1] / alpha[i-1] # compute values of alpha using our iterative formula
            alpha[i] = self._a[i] - beta[i-1] * self._c[i-1] # compute values of beta using our iterative formula

        self._alpha = alpha
        self._beta = beta 
        self._A_is_decomposed = True # Mark decomposition has been computed

    def lu_prod(self):
        """
        Method for multiplying the L and U factors obtained from our lu_decomp function above (using their sparse representation), to obtain the product LU as an instance of a tridiagonal matrix class

        Returns:
            TridiagonalMatrices instance representing LU
        """
        # Main diagonal: a_i = alpha_i + (beta_i * c_(i-1))
        lu_a = np.zeros(self.size)
        lu_a[0] = self._alpha[0] # first element has no contribution from beta or c
        lu_a[1:] = self._alpha[1:] + self._beta * self._c 

        # Lower diagonal: b_i = beta_i * alpha_(i-1)
        lu_b = self._beta * self._alpha[:-1]

        return TridiagonalMatrices(lu_a, lu_b, self._c) # Upper diagonal remains unchanged
    
    def subtract(self, other):
        """
        Subtract another entity, a class TridiagonalMatrices instance, from this one

        Arguments:
            other: TridiagonalMatrices instance of the same size

        Returns:
            TridiagonalMatrices instance representing self - other
        """

        return TridiagonalMatrices(
            self._a - other._a,
            self._b - other._b,
            self._c - other._c
        )
    
    def norm(self):
        """
        Compute the matrix norm of the sparse triadiagonal matrix.

        Returns:
            Matrix norm as a float
        """
        
        return np.sqrt(
            np.sum(self._a ** 2) +
            np.sum(self._b ** 2) +
            np.sum(self._c ** 2)
        )
    
    def error(self):
        """
        Compute the error ||A - LU|| using the sparse representation

        Returns:
            Matrix norm of A - LU as a float
        """
        lu = self.lu_prod()
        difference = self.subtract(lu)
        result = difference.norm()

        return result

    def solvelinearsystem(self,
                          b: np.ndarray) -> np.ndarray:
        """
        Solve the linear system Ax = b using LU decomposition

        Arguments:
            b: Right-hand side vector of length n (NumPy array)
            
        Returns:
            x: Solution vector of length n (NumPy array)
        """
        # Check is lu_decomp has been executed, to ensure only computing the LU decomposition once
        if not self._A_is_decomposed:
            self.lu_decomp()

        n = self.size
        y = np.zeros(n)
        x = np.zeros(n)

        # Forward substitution: Solve Ly = b. L has 1s on the main diagonal and self._beta on the lower diagonal, the equation is: y_i = b_i - beta(i-1) * y(i-1)
        y[0] = b[0]
        for i in range (1, n):
            y[i] = b[i] - self._beta[i-1] * y[i-1]

        # Backward substitution: Solve Ux = y. U has self._alpha on the main diagonal and self._c on the upper diagonal, the equation is: x_i = (y_i - c_i * x_(i+1)) / alpha_i
        x[n-1] = y[n-1] / self._alpha [n-1]
        for i in range(n-2, -1, -1):
            x[i] = (y[i] - self._c[i] * x[i+1]) / self._alpha[i]

        return x
    
    def power_iteration(self, tol = 1e-6, max_iter=1000):
        """
        Compute the largest eigenvalue with power iteration

        Starting from a random vector, apply the matvec_prod function repeatedly and normalise, until the eigenvalue estimate converges

        Arguments:
            tol: float, convergence tolerance (default = 1e-6)
            max_iter: int, maximum number of iterations (defaulf 1000)

        Returns:
            eigenvalue: float, estimate of the largest eigenvalue
        """

        # Create random vector and normalise it
        v = np.random.rand(self.size)
        v = v / np.sqrt(v @ v)

        eigenvalue = 0.0

        for i in range(max_iter):

            # Apply matrix-vector product
            w = self.matvec_prod(v)

            # Estimate eigenvalue
            new_eigenvalue = v @ w

            # Normalise to get the next vector
            v = w / np.sqrt(w @ w)

            # Check if it has converged within our tolerance
            if abs(new_eigenvalue - eigenvalue) < abs(tol):
                break

            eigenvalue = new_eigenvalue

        return eigenvalue
    
class FunctionDenoiser:
    """
    Samples a function on N+1 evenly distributed points in the interval [-1, 1], received standard deviation for noise, and stores both the noiseless and noisy samples
    """
    def __init__(self, N):
        """
        Initialise the denoiser with N intervals (N+1 points)

        Arguments:
            N : int, Number of intervals
        """

        self.N = N

        # Compute equidistant points x_0 = -1, ..., x_N = 1, step = 2/N

        self.x = np.array([-1 + 2 * i / N for i in range(N + 1)])

        # Q2.6 Build the structure of A once, so it depends only on N, not on beta or y
        n = N + 1
        self._A_diag = np.full(n, 2.0)
        self._A_diag[0] = self._A_diag[-1] = 1.0
        self._A_uplow = np.full(n-1, -1.0)

        # Initialise, set by add_noise()
        self.y_noiseless = None
        self.y = None

        # Initialise, set by smooth()
        self.s = None
        self._hessian = None # Q2.6 stores current I + beta*A
        self._current_beta = None # Q2.6 stores current beta



    def add_noise(self, 
                 g: callable,
                 sigma = 0.05 # set default standard deviation
                 ): 
        """
        Sample g at the discretisation points and add noise

        Arguments:
            g: callable, the exact function to be sampled. Must accept a NumPy array
            sigma: float, standard deviation of the measurement noise (default 0.05)
        """
        
        self.y_noiseless = g(self.x)
        epsilon = sigma * np.random.randn(self.N + 1)
        self.y = self.y_noiseless + epsilon

    def smooth(self, beta):
        """
        Constructs the tridiagonal matrix I + beta*A and solves the linear system using the lu_decomp and solvelinearsystem functions from TridiagonalMatrices, storing the smooth solution as an instance attribute

        Arguments:
            beta: float, smoothing parameter (>=0)
        """

        n = self.N + 1

        # Main diagonal, 1 at endpoints, 2 everywhere else
        a = np.full(n, 2.0)
        a[0], a[-1] = 1.0, 1.0

        # Upper and lower diagonals, -1 everywhere
        b = c = np.full(n-1, -1.0)

        # Compute I + beta * A: Main diagonal becomes 1 + beta * a, upper and lower diagonals become beta * (-1)
        main = 1.0 + beta * a
        lower = upper = beta * b # b = c so beta * b = beta * c

        A_matrix = TridiagonalMatrices(main, lower, upper)
        self.s = A_matrix.solvelinearsystem(self.y)

    def smooth_modified(self, beta):
        """
        Computationally efficient version of smooth()

        Reuses the Hessian I + beta*A and its LU decomposition if beta is not changed. It rebuilds them only when beta changes. Print messages indicate which steps are performed

        Arguments:
            beta: float, smoothing parameter (>=0)
        """
        
        print("Smoothing...")

        # Only recompute the Hessian and redo the LU decomposition if beta has changed
        if not beta == self._current_beta:
            print("Creating tridiagonal Hessian matrix...")
            main = 1.0 + beta * self._A_diag
            lower = upper = beta * self._A_uplow
            self._hessian = TridiagonalMatrices(main, lower, upper)
            self._current_beta = beta
            
            print("LU-decomposing...")
            self._hessian.lu_decomp()
        
        print("Solving linear system...")
        self.s = self._hessian.solvelinearsystem(self.y)

    def plot(self):
        """
        Plot the noiseless samples and the noisy measurements in a single graph
        """
        if self.y is None or self.y_noiseless is None:
            raise RuntimeError("Call add_noise() before plot()")
            

        plt.plot(self.x, self.y_noiseless, linewidth = 2, color = 'blue', label = "Original $g(x)$")
        plt.scatter(self.x, self.y, s = 15, marker = 'x', color = 'black', label = "Noisy measurements $y_i$")

        # Only plot smoothed solution if smooth() has been called
        if self.s is not None:
            plt.plot(self.x, self.s, linewidth = 2, color = 'red', label = "Smoothed $\\hat{s}$")
        
        plt.xlabel("$x$")
        plt.ylabel("$g(x)$")
        plt.title("Original vs noisy measurements")
        plt.legend()
        plt.show()

class FunctionDenoiserIter(FunctionDenoiser):
    """
    Smooths noisy 1D data, using gradient descent instead of LU decomposition. Inherits all functionalities from FunctionDenoiser, but overriding the smooth() and plot() functions.
    """

    def __init__(self, N):
        """
        Initialise the interative denoiser with N intervals (N+1) points

        Arguments:
            N: int, number of intervals
        """
        super().__init__(N)
        self._iterations = None # store iterations ran by smooth to use it in plot(). 3.5: Private, a counter only used for the plot title
    
    def copy_data(self, other):
        """
        Copy the noisy data from a FunctionDenoiser instance into this one, to make sure both methods (direct and iterative) use identical measurements.

        Arguments:
            other: FunctionDenoiser instance to copy data from
        """

        self.y = other.y.copy()
        self.y_noiseless = other.y_noiseless.copy()

    def smooth(self, beta, max_iter = 1000):
        """
        Smooth the noisy data using gradient

        Computes the step size alpha = 1 / lambda_max(I + beta * A) using power iteration, then iterates the gradient descent update formula:
            s = s - alpha * grad_f(s)
        where grad_f(s) = (I + beta * A)s - y

        Arguments:
            beta: float, smoothing parameter (>=0)
            max_iter: int, maximum number of gradient descent iterations (default 1000)
        """
        # Build I + beta * A as a TridiagonalMatrices instance
        diag = 1.0 + beta * self._A_diag
        uplow = beta * self._A_uplow
        self._hessian = TridiagonalMatrices(diag, uplow, uplow)

        # Step size: alpha = 1 / lambda_max(hessian)
        alpha = 1.0 / self._hessian.power_iteration()

        # Initialise s at the noisy measurements
        s = self.y.copy()

        # s = np.zeros(self.N) # to start from 0 (ignore, only for testing purposes)

        for i in range(max_iter):
            # Compute gradient: grad_f = (I + beta * A)s - y
            grad = self._hessian.matvec_prod(s) - self.y

            # Gradient descent update
            s = s - alpha * grad

        self.s = s
        self._iterations = max_iter # store number of iterations
        
    def plot(self):
        """
        Call the inherited plot() and add a supertitle that indicates the method and number of iterations used
        """
        plt.suptitle(f"Gradient descent method - {self._iterations} iterations")
        super().plot()

class ImageDenoiser:
    """
    Loads an image in colour, converts it to grayscale, adds noise, and denoises it using scipy's sparse solver. Reuses the Hessian when beta is unchanged.
    """

    def __init__(self, path="./", file="sonic.jpg"):
        """
        Load the image, convert to grayscale, and store it
        
        Arguments:
            path: str, directory where the image is located
            file: str, image filename (default "sonic.jpg")
        """
        from PIL import Image

        image = Image.open(path + file).convert("L") # Opens file, converts to L (for Luminance), a single brightness from 0 to 255 instead of RGB
        self._image = np.array(image)
        self._ny, self._nx = self._image.shape

        # set by add_noise() and smooth()
        self._y = None
        self._s = None
        self._hessian = None
        self._current_beta = None

    def add_noise(self, sigma = 2.0):
        """
        Add noise to the grayscale image and store it as a flat vector

        Arguments:
            sigma: float, noise standard deviation (default 2.0)
        """

        np.random.seed(1)
        
        noise = sigma * np.random.normal(0, 100, self._image.shape) # Creates 'map' of random numbers with same dimensions as our image, scale by standard deviation
        noisy = np.clip(self._image + noise, 0, 255).astype(np.uint8) # Adds noise to the image, ensuring no value is less than 0 or more than 255
        self._y = noisy.flatten().astype(float) # Turn 2d grid of pixels (image) into a 1d list (vector) of numbers

    def smooth(self, beta):
        """
        Denoise the image by solving (I + beta * C^T C)s = y

        Creates the Hessian H = I + beta * C^T C using Scipy sparse matrices, reusing it if beta has not changed. Solves the system with spsolve
        """
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        if self._y is None:
            raise RuntimeError("Call add_noise() before smooth()")
        
        # Only rebuild Hessian if beta has changed
        if beta != self._current_beta:
            
            print("Building new Hessian...")
            B = sparse.diags_array(
                (np.hstack((-np.ones(self._nx -1), 0.0)), np.ones(self._nx -1)), offsets = (0,1)
                ) # Create sparse matrix B, -1 on main diagonal with 0 as the last element, 1 on the upper diagonal, "offsets = (0,1)" tells first list to go on main diagonal (0) and second to go on upper diagonal (1)
            C = sparse.block_diag([B] * self._ny, format="csr") # Stacks N_y B matrices together, corner to corner, to form giant block diagonal matrix C. Compressed Sparse Row (csr) format so it only stores non-zero values, increasing efficieny.
            self._hessian = sparse.eye_array(self._nx * self._ny) + beta * C.T @ C # Build Hessian matrix, I + beta(C^T)C
            self._current_beta = beta
        
        print("Solving Linear system...")
        s = spsolve(self._hessian, self._y) # Solve linear system Hs = y
        self._s = np.reshape(s, (self._ny, self._nx)) # Reshape long pixels vector back into a 2d grid

    def plot(self):
        """
        Display the original grayscale image, the noisy image, and the smoothed image. Also shows a row and column grid of the smoothed result
        """
        
        if self._y is None:
            raise RuntimeError("Call add_noise before plot()")
        
        noisy_2d = np.reshape(self._y, (self._ny, self._nx)) # Reshape noisy data back into a 2d grid

        plt.gray()

        # Original, noisy and smoothed imaged
        fig, axes = plt.subplots(1, 3) # To see images side-by-side
        axes[0].imshow(self._image) # Display original image in first box
        axes[0].set_title("Original")
        axes[1].imshow(noisy_2d) # Display noisy image in the second box
        axes[1].set_title("Noisy")

        if self._s is not None:
            axes[2].imshow(self._s) # Display smoothed image in the third box
            axes[2].set_title(rf"Smoothed $\beta = {self._current_beta}$")

        plt.tight_layout() # to prevent overlap between images
        plt.show()

        # Since denoising was done by rows and not by columns, comparing a graph of rows/columns vs pixel intensity helps visualize the denoising
        if self._s is not None:
            plt.figure() 
            plt.title("Row 400") 
            plt.plot(self._s[400, :]) 
            plt.xlabel("Column index")
            plt.ylabel("Pixel intensity")
            plt.show() 

            plt.figure()
            plt.title("Column 300")
            plt.plot(self._s[:,300])
            plt.xlabel("Row index")
            plt.ylabel("Pixel intensity")
            plt.show()


        










       


    




     
        
    