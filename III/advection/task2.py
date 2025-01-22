# numerieke_methoden/III/advection/task2.py

class PhysConstants:
    def __init__(self):
        self.c      =        # advection velocity (m/s)
        self.Ag     =        # Gaussian wave amplitude ()
        self.sigmag =        # Gaussian wave width (m)
        self.Anot   =        # Molenkamp triangle height ()
        self.W      =        # Molenkamp triangle width (m)
# you may add your own constants if you wish


# 2) add here supporting code 




# 3) Use the following routine to start simulations
def Task2_caller(L, nx, TotalTime, dt, TimeSteppingMethod, Initialisation,
                 DiffMethod="CD"):
    # The mandatory input is:
    # L                   Length of domain to be modelled (m)
    # nx                  Number of gridpoint in the model domain
    # TotalTime           Total length of the simulation (s)
    # dt                  Length of each time step (s)
    # TimeSteppingMethod  Could be:
    #  "Theory"             Theoretical solution
    #  "AB"                 Adams-Bashforth
    #  "CN"                 Crank-Nicholson
    #  "EF"                 Euler Forward
    #  "LF"                 Leaf Frog
    #  "RK4"                Runge-Kutta 4
    # Initialization      Could be:
    #  "GaussWave"          Gauassian Wave
    #  "Molenkamp"          Molenkamp triangle
    #
    # The optional input is:
    # DiffMethod  Method to determine the 2nd order spatial derivative
    #   Default = "CD"    Central differences
    #    Option = "PS"    Pseudo spectral
    # 
    # The output is:
    # Time        a 1-D array (length nt) with time values considered
    # Xaxis       a 1-D array (length nx) with x-values used
    # Result      a 2-D array (size [nx, nt]), with the results of the routine    
    # You may add extra output after these three
    
    PhysC = PhysConstants()       # load physical constants in `self`-defined variable PhysC
    # (start code)
    
    return Time, Xaxis, Result    


# 4) Code here (or in another file) the commands that use Task2_caller

# 5) Code here (or in another file) the analysis of the results and the plotting
