import numpy as np

# global constant
hbarc = 0.197326938 

# structs/classes to store simulation variables
class hydro_parameters:
    def __init__(self,
                tau_initial,
                energy_initial,
                shear_initial,
                freezeout_temperature_GeV,
                energy_min,
                flux_limiter,
                regulation_scheme,
                dynamical_variables,
                temperature_etas,
                eta_bar,
                constant_etas,
                etas_min,
                etas_aL,
                etas_aH,
                etas_etask,
                etas_Tk_GeV):
        self.tau_initial               = tau_initial
        self.energy_initial            = energy_initial
        self.shear_initial             = shear_initial
        self.freezeout_temperature_GeV = freezeout_temperature_GeV
        self.energy_min                = energy_min
        self.flux_limiter              = flux_limiter
        self.regulation_scheme         = regulation_scheme
        self.dynamical_variables       = dynamical_variables
        self.temperature_etas          = temperature_etas
        self.eta_bar                   = eta_bar
        self.constant_etas             = constant_etas
        self.etas_min                  = etas_min
        self.etas_aL                   = etas_aL
        self.etas_aH                   = etas_aH
        self.etas_etask                = etas_etask
        self.etas_Tk_GeV               = etas_Tk_GeV

# ---------------------------------------------------------------------
class lattice_parameters:
    def __init__(self,
                lattice_points_x,
                lattice_points_y,
                lattice_points_eta,
                lattice_spacing_x,
                lattice_spacing_y,
                lattice_spacing_eta,
                adaptive_time_step,
                delta_0,
                alpha,
                fixed_time_step,
                max_time_steps):
        self.lattice_points_x    = lattice_points_x
        self.lattice_points_y    = lattice_points_y
        self.lattice_points_eta  = lattice_points_eta
        self.lattice_spacing_x   = lattice_spacing_x
        self.lattice_spacing_y   = lattice_spacing_y
        self.lattice_spacing_eta = lattice_spacing_eta
        self.adaptive_time_step  = adaptive_time_step
        self.delta_0             = delta_0
        self.alpha               = alpha
        self.fixed_time_step     = fixed_time_step
        self.max_time_steps      = max_time_steps

# class to run hydro simulation
class Hydro_Sim:
    def __init__(self, hydro, lattice):
        '''
        hydro - instance of class (struct) containing all the hydro parameters
        lattice = instance of class containg all lattice parameters
        '''
        self.m_hydro = hydro
        self.m_lattice = lattice
    
    # --------------------------------------------------------------
    # hyrdo variables container
    class hydro_variables:
            def __init__(self, size):    
                self.m_size = size
                
                # time-like energy-momentum tensor components
                self.Ttt = np.zeros(size, dtype=float)
                self.Ttx = np.zeros(size, dtype=float)
                self.Tty = np.zeros(size, dtype=float)
                self.Ttn = np.zeros(size, dtype=float)
                
                # shear-stress components
                self.pitt = np.zeros(size, dtype=float)
                self.pitx = np.zeros(size, dtype=float)
                self.pity = np.zeros(size, dtype=float)
                self.pitn = np.zeros(size, dtype=float)
                self.pixx = np.zeros(size, dtype=float)
                self.piyy = np.zeros(size, dtype=float)
                self.pixy = np.zeros(size, dtype=float)
                self.pixn = np.zeros(size, dtype=float)
                self.piyn = np.zeros(size, dtype=float)
                self.pinn = np.zeros(size, dtype=float)
                
    def swap_hydro_variables(self, hydro_A, hydro_B):
        # maybe add flag to check hydro_variables classes have same size
        temp = self.hydro_variables(hydro_A.m_size)
        temp = hydro_A
        hydro_A = hydro_B
        hydro_B = temp
        del temp
    
    # ------------------------------------------------------------------
    # fluid velocity container
    class fluid_velocity:
        def __init__(self, size):
            self.m_size = size
            
            self.ux = np.zeros(size, dtype=float)
            self.uy = np.zeros(size, dtype=float)
            self.un = np.zeros(size, dtype=float)
            
    def swap_fluid_velocity(self, FV_A, FV_B):
        temp = self.fluid_velocity(FV_A.m_size)
        FV_A = FV_B
        FV_B = temp
        del temp
    
    # ------------------------------------------------------------------
    def setup_lattice(self):
        '''
        Sets up variables to make grid
        '''
        Nx   = self.m_lattice.lattice_points_x
        Ny   = self.m_lattice.lattice_points_y
        Neta = self.m_lattice.lattice_points_eta
        
        # add 4 to account for ghost and white cells
        Nx   += 4
        Ny   += 4
        Neta += 4
        
        # length of array for all celss
        self.m_size = Nx * Ny * Neta
        
        # hydro quantities
        self.e  = np.zeros(self.m_size, dtype=float)
    
    # ------------------------------------------------------------------    
    def allocate_dynamical_variable(self):
        '''
        The name says it all
        '''
        # fluid velocity variables
        self.u   = self.fluid_velocity(self.m_size)
        self.u_p = self.fluid_velocity(self.m_size)
        
        # conserved current variables
        self.q  = self.hydro_variables(self.m_size)
        self.Q  = self.hydro_variables(self.m_size)
        self.qI = self.hydro_variables(self.m_size)
    
    # ------------------------------------------------------------------    
    def calc_linear_array_index(self, i, j, k, nx, ny):
        '''
        Since grid is 3 dimensional, but hydro variables and fluid velocity
        is 1-d array, we need to translate between indices
        '''
        return i + nx * (j + ny * k)
    
    # ------------------------------------------------------------------    
    def calc_conformal_energy(self, T_switch):
        '''
        Calculates the energy density give a temperature,
        used primarily for calculating the freezeout energy density
        '''
        def conformal_factor():
            colors = 3.
            flavors = 3.
            return np.pi ** 2 * (2 * (colors ** 2 - 1) + 3.5 * colors * flavors) / 30.
        return conformal_factor() * T_switch ** 4 
        
    # ------------------------------------------------------------------    
    def check_all_cells_below_freezeout(self):
        '''
        Function that checks if simulation is done,
        condition: all energy density cells are below the switching temperature
        '''
        all_below = False
        n = 0
        e_switch = self.calc_conformal_energy(self.m_hydro.freezeout_temperature_GeV / hbarc)
        for e in self.e:
            if e > e_switch:
                return False
        return True
    
    # ------------------------------------------------------------------    
    def set_ghost_cell_BC(self, s, sBC):
        '''
        This function matches the ghost cell boundary condition to the
        phyiscal boundary conditions
        s - is the ghost cell index
        sBC - is the physical boundary condition index
        '''
        self.e[s] = self.e[sBC]
        
        self.u.ux[s] = self.u.ux[sBC]
        self.u.uy[s] = self.u.uy[sBC]
        self.u.un[s] = self.u.un[sBC]
        
        self.q.Ttt[s] = self.q.Ttt[sBC]
        self.q.Ttx[s] = self.q.Ttx[sBC]
        self.q.Tty[s] = self.q.Tty[sBC]
        self.q.Ttn[s] = self.q.Ttn[sBC]
        
        self.q.pitt[s] = self.q.pitt[sBC] 
        self.q.pitx[s] = self.q.pitx[sBC]
        self.q.pity[s] = self.q.pity[sBC]
        self.q.pitn[s] = self.q.pitn[sBC]
        self.q.pixx[s] = self.q.pixx[sBC]
        self.q.piyy[s] = self.q.piyy[sBC]
        self.q.pixy[s] = self.q.pixy[sBC]
        self.q.pixn[s] = self.q.pixn[sBC]
        self.q.piyn[s] = self.q.piyn[sBC]
        self.q.pinn[s] = self.q.pinn[sBC]
        
    # ------------------------------------------------------------------    
    def set_ghost_cells(self):
        '''
        This function loops throught the boundaries and calls the function
        set_ghost_cell_BC to set the ghost cell values
        '''
        Nx   = self.m_lattice.lattice_points_x
        Ny   = self.m_lattice.lattice_points_y
        Neta = self.m_lattice.lattice_points_eta
        
        # loop over y,z boundary
        for j in range(2, Ny +2):
            for k in range(2, Neta + 2):
                for i in range(2):  # left ghost cells
                    s   = self.calc_linear_array_index(i, j, k, Nx + 4, Ny + 4)
                    sBC = self.calc_linear_array_index(2, j, k, Nx + 4, Ny + 4)
                    self.set_ghost_cell_BC(s, sBC)
                for i in range(Nx + 2, Nx + 4):   # right ghost cells
                    s   = self.calc_linear_array_index(i, j, k, Nx + 4, Ny + 4)
                    sBC = self.calc_linear_array_index(Nx + 1, j, k, Nx + 4, Ny + 4)
                    self.set_ghost_cell_BC(s, sBC)
                    
        # loop over x,z boundary
        for i in range(2, Nx + 2):
            for k in range(2, Neta + 2):
                for j in range(2):  # front ghost cells
                    s   = self.calc_linear_array_index(i, j, k, Nx + 4, Ny + 4)
                    sBC = self.calc_linear_array_index(i, 2, k, Nx + 4, Ny + 4)
                    self.set_ghost_cell_BC(s, sBC)
                for j in range(Ny + 2, Ny + 4): # back ghost cells
                    s   = self.calc_linear_array_index(i, j, k, Nx + 4, Ny + 4)
                    sBC = self.calc_linear_array_index(i, Ny + 1, k, Nx + 4, Ny + 4)
                    self.set_ghost_cell_BC(s, sBC)
                    
        # loop over x,y boundary
        for i in range(2, Nx + 2):
            for j in range(2, Neta + 2):
                for k in range(2):  # front ghost cells
                    s   = self.calc_linear_array_index(i, j, k, Nx + 4, Ny + 4)
                    sBC = self.calc_linear_array_index(i, k, 2, Nx + 4, Ny + 4)
                    self.set_ghost_cell_BC(s, sBC)
                for k in range(Neta + 2, Neta + 4): # back ghost cells
                    s   = self.calc_linear_array_index(i, j, k, Nx + 4, Ny + 4)
                    sBC = self.calc_linear_array_index(i, j, Neta + 1, Nx + 4, Ny + 4)
                    self.set_ghost_cell_BC(s, sBC)
    
    # ------------------------------------------------------------------    
    def set_BC(self, model, tau):
        '''
        Since the background solution is not zero, i.e. we have Bjorken 
        expansion. However, the background fields are only a function of 
        proper time.
        This function initializes the grid to the background value
        '''
        # get background field value for current time step
        veps_, pi_ = model.background_evolution(veps_0=self.m_hydro.energy_initial,
                                                pi_0=self.m_hydro.shear_initial,
                                                t_0=self.m_hydro.tau_initial,
                                                t=tau)
        def set_hydro_vars(sBC, veps, pi):
            '''
            Utility function for setting boundary conditions
            '''
            self.e[sBC] = veps

            self.u.ux[sBC] = 0.
            self.u.uy[sBC] = 0.
            self.u.un[sBC] = 0.

            self.q.Ttt[sBC] = veps
            self.q.Ttx[sBC] = 0.
            self.q.Tty[sBC] = 0.
            self.q.Ttn[sBC] = 0.

            self.q.pitt[sBC] = 0. 
            self.q.pitx[sBC] = 0.
            self.q.pity[sBC] = 0.
            self.q.pitn[sBC] = 0.
            self.q.pixx[sBC] = pi / 2.
            self.q.piyy[sBC] = pi / 2.
            self.q.pixy[sBC] = 0.
            self.q.pixn[sBC] = 0.
            self.q.piyn[sBC] = 0.
            self.q.pinn[sBC] = -pi / tau ** 2
            
        Nx   = self.m_lattice.lattice_points_x
        Ny   = self.m_lattice.lattice_points_y
        Neta = self.m_lattice.lattice_points_eta
        
        # set the rest of the grid 
        for k in range(2, Neta + 2):
            for j in range(2, Ny + 2):
                for i in range(2, Nx + 2):
                    # linear array index
                    s = self.calc_linear_array_index(i, j, k, Nx + 4, Ny + 4)
                    set_hydro_vars(s, veps_, pi_)
    
    # ------------------------------------------------------------------
    def get_energy_density_neighbors(self, sim, sip, sjm, sjp, skm, skp):
        '''
        The name is pretty clear
        -----------
        e1       - energy denisty neigbors (return value)
        sim, sip - neighbors in x indices
        sjm, sjp - neighbors in y indices
        skm, skp - neighbors in y indices
        '''
        e1 = np.zeros(6)
        e1[0] = self.e[sim]
        e1[1] = self.e[sip]
        e1[2] = self.e[sjm]
        e1[3] = self.e[sjp]
        e1[4] = self.e[skm]
        e1[5] = self.e[skp]
        return e1
    
    # ------------------------------------------------------------------
    def get_fluid_velocity_neighbors(self, simm, sim, sip, sipp, sjmm, sjm, sjp, sjpp, skmm, skm, skp, skpp, t2):
        '''
        The name is pretty clear
        -------------
        u            - fluid velocity element
        s            - linear array index
        i, j, k      - x, y, n
        mm, m, p, pp - index minus 2, index minus 1, index plus 1, index plus 2
        t2           - proper time squared
        ui1, uj1, uk1 - contain first order neighbors in (x,y,n) grid
        vxi, vyj, vnk - contain second order neighbors for spatial velocities
        --

        (ui1, uj1, uk1, vxi, vyj, vnk) - returned tuple
        '''
        # ------
        # x-grid variable setup
        ux_simm = self.u.ux[simm]
        uy_simm = self.u.uy[simm]
        un_simm = self.u.un[simm]
        ut_simm = np.sqrt(1 + ux_simm ** 2 + uy_simm ** 2 + t2 * un_simm ** 2)

        ux_sim = self.u.ux[sim]
        uy_sim = self.u.uy[sim]
        un_sim = self.u.un[sim]
        ut_sim = np.sqrt(1 + ux_sim ** 2 + uy_sim ** 2 + t2 * un_sim ** 2)

        ux_sip = self.u.ux[sip]
        uy_sip = self.u.uy[sip]
        un_sip = self.u.un[sip]
        ut_sip = np.sqrt(1 + ux_sip ** 2 + uy_sip ** 2 + t2 * un_sip ** 2)

        ux_sipp = self.u.ux[sipp]
        uy_sipp = self.u.uy[sipp]
        un_sipp = self.u.un[sipp]
        ut_sipp = np.sqrt(1 + ux_sipp ** 2 + uy_sipp ** 2 + t2 * un_sipp ** 2)

        # construct spatial x-velocities and set neighbors
        vxi = np.zeros(4)
        vxi[0] = ux_simm / ut_simm
        vxi[1] = ux_sim / ut_sim
        vxi[2] = ux_sip / ut_sip
        vxi[3] = ux_sipp / ut_sipp

        # neighboring x-grid fluid velocities
        ui1 = np.zeros(6)
        ui1[0] = ux_sim
        ui1[1] = ux_sip
        ui1[2] = uy_sim
        ui1[3] = uy_sip
        ui1[4] = un_sim
        ui1[5] = un_sip

        # ------
        # y-grid variable setup
        ux_sjmm = self.u.ux[sjmm]
        uy_sjmm = self.u.uy[sjmm]
        un_sjmm = self.u.un[sjmm]
        ut_sjmm = np.sqrt(1 + ux_sjmm ** 2 + uy_sjmm ** 2 + t2 * un_sjmm ** 2)

        ux_sjm = self.u.ux[sjm]
        uy_sjm = self.u.uy[sjm]
        un_sjm = self.u.un[sjm]
        ut_sjm = np.sqrt(1 + ux_sjm ** 2 + uy_sjm ** 2 + t2 * un_sjm ** 2)

        ux_sjp = self.u.ux[sjp]
        uy_sjp = self.u.uy[sjp]
        un_sjp = self.u.un[sjp]
        ut_sjp = np.sqrt(1 + ux_sjp ** 2 + uy_sjp ** 2 + t2 * un_sjp ** 2)

        ux_sjpp = self.u.ux[sjpp]
        uy_sjpp = self.u.uy[sjpp]
        un_sjpp = self.u.un[sjpp]
        ut_sjpp = np.sqrt(1 + ux_sjpp ** 2 + uy_sjpp ** 2 + t2 * un_sjpp ** 2)

        # construct spatial y-velocities and set neighbors
        vyj = np.zeros(4)
        vyj[0] = uy_sjmm / ut_sjmm
        vyj[1] = uy_sjm / ut_sjm
        vyj[2] = uy_sjp / ut_sjp
        vyj[3] = uy_sjpp / ut_sjpp

        # neighboring y-grid fluid velocities
        uj1 = np.zeros(6)
        uj1[0] = ux_sjm
        uj1[1] = ux_sjp
        uj1[2] = uy_sjm
        uj1[3] = uy_sjp
        uj1[4] = un_sjm
        uj1[5] = un_sjp

        # -------
        # x-grid variable setup
        ux_skmm = self.u.ux[skmm]
        uy_skmm = self.u.uy[skmm]
        un_skmm = self.u.un[skmm]
        ut_skmm = np.sqrt(1 + ux_skmm ** 2 + uy_skmm ** 2 + t2 * un_skmm ** 2)

        ux_skm = self.u.ux[skm]
        uy_skm = self.u.uy[skm]
        un_skm = self.u.un[skm]
        ut_skm = np.sqrt(1 + ux_skm ** 2 + uy_skm ** 2 + t2 * un_skm ** 2)

        ux_skp = self.u.ux[skp]
        uy_skp = self.u.uy[skp]
        un_skp = self.u.un[skp]
        ut_skp = np.sqrt(1 + ux_skp ** 2 + uy_skp ** 2 + t2 * un_skp ** 2)

        ux_skpp = self.u.ux[skpp]
        uy_skpp = self.u.uy[skpp]
        un_skpp = self.u.un[skpp]
        ut_skpp = np.sqrt(1 + ux_skpp ** 2 + uy_skpp ** 2 + t2 * un_skpp ** 2)

        # construct spatial velocities and set neighbors
        vnk = np.zeros(4)
        vnk[0] = ux_skmm / ut_skmm
        vnk[1] = ux_skm / ut_skm
        vnk[2] = ux_skp / ut_skp
        vnk[3] = ux_skpp / ut_skpp

        # neighboring x-grid fluid velocities
        uk1 = np.zeros(6)
        uk1[0] = ux_skm
        uk1[1] = ux_skp
        uk1[2] = uy_skm
        uk1[3] = uy_skp
        uk1[4] = un_skm
        uk1[5] = un_skp
        return ui1, uj1, uk1, vxi, vyj, vnk
    
    # ------------------------------------------------------------------
    def get_hydro_variables_neighbors(self, sm, sp):
        '''
        The name is pretty clear
        ---------
        qm - stores i-1 neighbor
        qp - stores i+1 neighbor
        q  - dynamical variable returned by function (return value)
        '''
        N = self.m_hydro.dynamical_variables
        q = np.zeros(2 * N)

        i = 0
        q[i] = self.q.Ttt[sm]; i += 1
        q[i] = self.q.Ttt[sp]; i += 1

        q[i] = self.q.Ttx[sm]; i += 1
        q[i] = self.q.Ttx[sp]; i += 1

        q[i] = self.q.Tty[sm]; i += 1
        q[i] = self.q.Tty[sp]; i += 1

        q[i] = self.q.Ttn[sm]; i += 1
        q[i] = self.q.Ttn[sp]; i += 1

        q[i] = self.q.pitt[sm]; i += 1
        q[i] = self.q.pitt[sp]; i += 1

        q[i] = self.q.pitx[sm]; i += 1
        q[i] = self.q.pitx[sp]; i += 1

        q[i] = self.q.pity[sm]; i += 1
        q[i] = self.q.pity[sp]; i += 1

        q[i] = self.q.pitn[sm]; i += 1
        q[i] = self.q.pitn[sp]; i += 1

        q[i] = self.q.pixx[sm]; i += 1
        q[i] = self.q.pixx[sp]; i += 1

        q[i] = self.q.pixy[sm]; i += 1
        q[i] = self.q.pixy[sp]; i += 1

        q[i] = self.q.pixn[sm]; i += 1
        q[i] = self.q.pixn[sp]; i += 1

        q[i] = self.q.piyy[sm]; i += 1
        q[i] = self.q.piyy[sp]; i += 1

        q[i] = self.q.piyn[sm]; i += 1
        q[i] = self.q.piyn[sp]; i += 1

        q[i] = self.q.pinn[sm]; i += 1
        q[i] = self.q.pinn[sp]
        return q
    
    # ------------------------------------------------------------------
    def minmod(self, x, y):
        return (np.sign(x) + np.sign(y)) * (np.fmin(np.fabs(x), np.fabs(y))) / 2

    def minmod3(self, x, y, z):
        return self.minmod(x, self.minmod(y, z))

    def minmod_derivative(self, q, qp, qm):
        theta = self.m_hydro.flux_limiter
        return self.minmod3(theta * (q - qm), (qp - qm) / 2, theta * (qp - q))
    
    def compute_max_local_propogation_speed(self, v_data, v):
        # neigboring velocities
        vmm = v_data[0]
        vm  = v_data[1]
        vp  = v_data[2]
        vpp = v_data[3]

        # velocity derivatves
        dvp = self.minmod_derivative(q=vp, qp=vpp, qm=v)
        dv  = self.minmod_derivative(q=v, qp=vp, qm=vm)
        dvm = self.minmod_derivative(q=vm, qp=v, qm=vmm)

        #extrapolated velocities
        vRp = vp - dvp / 2   # v^{+}_{a+1/2}
        vRm = v - dv / 2     # v^{+}_{a-1/2}
        vLp = v + dv / 2     # v^{-}_{a+1/2}
        vLm = vm + dvm / 2   # v^{-}_{a-1/2}

        # local propogation speeds
        sp = np.fmax(np.fabs(vLp), np.fabs(vRp))  # s_{a+1/2}
        sm = np.fmax(np.fabs(vLm), np.fabs(vRm))  # s_{a-1/2}
        return sm, sp
    
    # ------------------------------------------------------------------    
    def flux_terms(self, q_data, q1_data, q2_data, v_data, v):
        '''
        This function computes the flux terms needed to compute the Kurganov-Tadmor
        time-step. The flux refers to the numerical flux at the faces of voxel in
        our simulation
        -----------
        The naming of the variables follows from the equations in the paper Eqs. (44)-(53)
        q - <Ttt, Ttx, Tty, Ttn, pitt, pitt, pitx, pity, pitn, pixx, pinn, piyy, pixy, pixn, piyn>
        --- 
        returns tuple (Hm, Hp) which are arrays for the numerical fluxes for each dynamic variable
        '''   
        # local propogation speed
        sm, sp = self.compute_max_local_propogation_speed(v_data, v)
        
        # neigboring velocities
        vmm = v_data[0]
        vm  = v_data[1]
        vp  = v_data[2]
        vpp = v_data[3]

        # velocity derivatves
        dvp = self.minmod_derivative(q=vp, qp=vpp, qm=v)
        dv  = self.minmod_derivative(q=v, qp=vp, qm=vm)
        dvm = self.minmod_derivative(q=vm, qp=v, qm=vmm)

        #extrapolated velocities
        vRp = vp - dvp / 2   # v^{+}_{a+1/2}
        vRm = v - dv / 2     # v^{+}_{a-1/2}
        vLp = v + dv / 2     # v^{-}_{a+1/2}
        vLm = vm + dvm / 2   # v^{-}_{a-1/2}
        
        # Compute flux terms Hp, H,
        N = len(q_data)
        Hm = np.zeros(N)
        Hp = np.zeros(N)
        for i, j in zip(range(N), range(0, 2*N, 2)):
            '''
            i - the current voxel
            j - the neighboring voxel
            '''
            # same process as above, but with the dynamical variables
            q   = q_data[i]       # current fluid cell
            qmm = q2_data[j]      # cell i-2
            qm  = q1_data[j]      # cell i-1
            qp  = q1_data[j + 1]  # cell i+1
            qpp = q2_data[j + 1]  # cell i+2
            
            dqp = self.minmod_derivative(q=qp, qp=qpp, qm=q)
            dq  = self.minmod_derivative(q=q, qp=qp, qm=qm)
            dqm = self.minmod_derivative(q=qm, qp=q, qm=qmm)
            
            qRp = qp - dqp / 2   # q^{+}_{a+1/2}
            qRm = q - dq / 2     # q^{+}_{a-1/2}
            qLp = q + dq / 2     # q^{-}_{a+1/2}
            qLm = qm + dqm / 2   # q^{-}_{a-1/2}
            
            # fluid currents q * v
            Fm = qm * vm
            F  = q * v 
            Fp = qp * vp
            
            # extrapolated currents
            FRp = Fp - (qp * dvp + vp * dqp) / 2  # F^{+}_{a+1/2}
            FRm = F - (q * dv + v * dq) / 2       # F^{+}_{a-1/2}
            FLp = F + (q * dv + v * dq) / 2       # F^{-}_{a+1/2}
            FLm = Fm + (qm * dvm + vm * dqm) / 2  # F^{-}_{a-1/2}
            
            # numericals fluxes for FT algo
            Hp[i] = (FRp + FLp - sp * (qRp - qLp)) / 2   # H_{a+1/2}
            Hm[i] = (FRm + FLm - sm * (qRm - qLm)) / 2   # H_{a-1/2}
        return Hm, Hp
    
    # ------------------------------------------------------------------    
    def calculate_euler_step(self, tau, dt, dt_prev, update, RK2, model):
        '''
        This function calculates the first Euler step in the RK2-KT algo
        ----------------
        tau   - proper time for current step
        dt         - adaptive time step size
        dt_prev    - last adaptive time step size
        update     - flag for which update to use
        RK2        - flag for which RK step we are one
        model - contains the functions that calculate the source terms 
        ----------------
        The important variables for this function are brieflu summarized in this 
        comment, but repeated throughout the code again, as this is the most 
        involved function
        Dynamical variables typically include: 
        Ttt, Ttx, Tty, Ttn, pitt, pitt, pitx, pity, pitn, pixx, pinn, piyy, pixy, pixn, piyn
        and maybe more
        qs                - array containing the dynamical variables for cell s
        E                 - he total source term array used to do the Euler update
        S                 - external source term arry used in 'E' arrat above
        qi1, qi2          - dynamical variables arrays for 1st and 2nd order neighbor in x
        qj1, qj2          - "         "         "      "   "   "   "   "     "        "  y
        qk1, qk2          - "         "         "      "   "   "   "   "     "        "  z
        e1                - 1st order neighbors for energy denisty
        ui1, uj1, uk1     - 1st and 2nd order neighbors for fluid velocity
        vxi, vyj, vnk     - 1st and 2nd order neighbors for spatial velocity
        Hx_plus, Hx_minus - x numerical flux terms
        Hy_plus, Hy_minus - y numerical flux terms
        Hz_plus, Hz_minus - z numerical flux terms
        '''        
        # the meat and potatoes of this function starts now
        Nx   = self.m_lattice.lattice_points_x
        Ny   = self.m_lattice.lattice_points_y
        Neta = self.m_lattice.lattice_points_eta
        
        dx = self.m_lattice.lattice_spacing_x
        dy = self.m_lattice.lattice_spacing_y
        deta = self.m_lattice.lattice_spacing_eta
        
        theta = self.m_hydro.flux_limiter
        tau2  = tau * tau
        
        # variable for corresponding y indices in linear array
        stride_y = Nx + 4
        # variable for corresponding eta indices in linear array
        stride_eta = (Nx + 4) * (Ny + 4)
        
        N_dynamic = self.m_hydro.dynamical_variables
        # loop over cells (voxels) in simulation 
        for k in range(2, Neta + 2):
            for j in range(2, Nx + 2):
                for i in range(2, Neta + 2):
                    qs = np.zeros(N_dynamic)    # dynamical variables in cell s
                    E  = np.zeros(N_dynamic)    # total source term for update
                    S  = np.zeros(N_dynamic)    # external source term for update
                    
                    qi1 = np.zeros(2 * N_dynamic)   # +/- 1 neighbor of qs in x
                    qj1 = np.zeros(2 * N_dynamic)   # +/- 1 neighbor of qs in y
                    qk1 = np.zeros(2 * N_dynamic)   # +/- 1 neighbor of qs in eta
                    
                    qi2 = np.zeros(2 * N_dynamic)   # +/- 2 neighbor of qs in x
                    qj2 = np.zeros(2 * N_dynamic)   # +/- 2 neighbor of qs in y
                    qk2 = np.zeros(2 * N_dynamic)   # +/- 2 neighbor of qs in eta
                    
                    e1  = np.zeros(6)   # energy density for {i-1,i+1,j-1,j+1,k-1,k+1}
                    
                    ui1 = np.zeros(6)   # fluid velocity for all 1-neighbors in x
                    uj1 = np.zeros(6)   # fluid velocity for all 1-neighbors in y
                    uk1 = np.zeros(6)   # fluid velocity for all 1-neighbors in eta
                    
                    vxi = np.zeros(4)   # vx for {i-2,i-1,i+1,i+2}
                    vyj = np.zeros(4)   # vy for {j-2,j-1,j+1,j+2}
                    vnk = np.zeros(4)   # vn for {k-2,k-1,k+1,k+2}
                    
                    Hx_plus  = np.zeros(N_dynamic)   # numerical flux on right x-face
                    Hx_minus = np.zeros(N_dynamic)   # numerical flux on left x-face
                    Hy_plus  = np.zeros(N_dynamic)   # numerical flux on right y-face
                    Hy_minus = np.zeros(N_dynamic)   # numerical flux on left y-face
                    Hn_plus  = np.zeros(N_dynamic)   # numerical flux on right eta-face
                    Hn_minus = np.zeros(N_dynamic)   # numerical flux on left eta-face
                    
                    # index for linear 3d array
                    s = self.calc_linear_array_index(i, j, k, Nx + 4, Ny + 4)
                    
                    # neighboring x indices
                    simm = s - 2
                    sim  = s - 1
                    sip  = s + 1 
                    sipp = s + 2
                    
                    # neighboring y indices
                    sjmm = s - 2 * stride_y
                    sjm  = s - stride_y
                    sjp  = s + stride_y
                    sjpp = s + 2 * stride_y
                    
                    # neighboring eta indices
                    skmm = s - 2 * stride_eta
                    skm  = s - stride_eta
                    skp  = s + stride_eta
                    skpp = s + 2 * stride_eta 
                    
                    # set dynamical variabels for qs
                    i = 0
                    qs[i] = self.q.Ttt[s]; i += 1
                    qs[i] = self.q.Ttx[s]; i += 1
                    qs[i] = self.q.Tty[s]; i += 1
                    qs[i] = self.q.Ttn[s]; i += 1
                    qs[i] = self.q.pitt[s]; i += 1
                    qs[i] = self.q.pitx[s]; i += 1
                    qs[i] = self.q.pity[s]; i += 1
                    qs[i] = self.q.pitn[s]; i += 1
                    qs[i] = self.q.pixx[s]; i += 1
                    qs[i] = self.q.pixy[s]; i += 1
                    qs[i] = self.q.pixn[s]; i += 1
                    qs[i] = self.q.piyy[s]; i += 1
                    qs[i] = self.q.piyn[s]; i += 1
                    qs[i] = self.q.pinn[s]
                    
                    # set fluid velocities
                    ux = self.u.ux[s]
                    uy = self.u.uy[s]
                    un = self.u.un[s]
                    ut = np.sqrt(1 + ux ** 2 + uy ** 2 + tau2 * un ** 2)
                    
                    # set previous fluid velocities
                    ux_p = self.u_p.ux[s]
                    uy_p = self.u_p.uy[s]
                    un_p = self.u_p.un[s]
                    
                    # energy density in cell s
                    e_s = self.e[s]
                    
                    # calculate numerical spatial derivatives
                    e1 = self.get_energy_density_neighbors(sim, sip, sjm, sjp, skm, skp)
                    ui1, uj1, uk1, vxi, vyj, vnk = self.get_fluid_velocity_neighbors(simm, sim, sip, sipp, sjmm, sjm, sjp, sjpp, skmm, skm, skp, skpp, tau2)
                    qi1 = self.get_hydro_variables_neighbors(sim, sip)
                    qi2 = self.get_hydro_variables_neighbors(simm, sipp)
                    qj1 = self.get_hydro_variables_neighbors(sjm, sjp)
                    qj2 = self.get_hydro_variables_neighbors(sjmm, sjpp)
                    qk1 = self.get_hydro_variables_neighbors(skm, skp)
                    qk2 = self.get_hydro_variables_neighbors(skmm, skpp)
                    
                    # calculate source term
                    S = model.source_ternms(q=qs, e=e_s, t=tau, 
                                            qi1=qi1, qj1=qj1, qk1=qk1, 
                                            e1=e1, 
                                            ui1=ui1, uj1=uj1, uk1=uk1, 
                                            ux=ux, uy=uy, un=un, 
                                            ux_p=ux_p, uy_p=uy_p, un_p=un_p, 
                                            dt_prev=dt_prev, dx=dx, dy=dy, deta=deta, 
                                            hydro=self.m_hydro)
                    
                    # calculate flux terms
                    Hx_minus, Hx_plus = self.flux_terms(qs, qi1, qi2, vxi, ux / ut)
                    Hy_minus, Hy_plus = self.flux_terms(qs, qj1, qj2, vyj, uy / ut)
                    Hn_minus, Hn_plus = self.flux_terms(qs, qk1, qk2, vnk, un / ut)
                    
                    # store results 
                    if not update:
                        # store source function in qI
                        E = S + (Hx_minus - Hx_plus) / dx + (Hy_minus - Hy_plus) / dy + (Hn_minus - Hn_plus) / deta
                        i = 0
                        self.qI.Ttt[s]  = E[i]; i += 1
                        self.qI.Ttx[s]  = E[i]; i += 1
                        self.qI.Tty[s]  = E[i]; i += 1
                        self.qI.Ttn[s]  = E[i]; i += 1
                        self.qI.pitt[s] = E[i]; i += 1
                        self.qI.pitx[s] = E[i]; i += 1
                        self.qI.pity[s] = E[i]; i += 1
                        self.qI.pitn[s] = E[i]; i += 1
                        self.qI.pixx[s] = E[i]; i += 1
                        self.qI.pixy[s] = E[i]; i += 1
                        self.qI.pixn[s] = E[i]; i += 1
                        self.qI.piyy[s] = E[i]; i += 1
                        self.qI.piyn[s] = E[i]; i += 1
                        self.qI.pinn[s] = E[i]
                    elif not RK2:
                        # store q + E.dt in qI
                        qs += dt * (S + (Hx_minus - Hx_plus) / dx + (Hy_minus - Hy_plus) / dy + (Hn_minus - Hn_plus) / deta)
                        i = 0
                        self.qI.Ttt[s]  = qs[i]; i += 1
                        self.qI.Ttx[s]  = qs[i]; i += 1
                        self.qI.Tty[s]  = qs[i]; i += 1
                        self.qI.Ttn[s]  = qs[i]; i += 1
                        self.qI.pitt[s] = qs[i]; i += 1
                        self.qI.pitx[s] = qs[i]; i += 1
                        self.qI.pity[s] = qs[i]; i += 1
                        self.qI.pitn[s] = qs[i]; i += 1
                        self.qI.pixx[s] = qs[i]; i += 1
                        self.qI.pixy[s] = qs[i]; i += 1
                        self.qI.pixn[s] = qs[i]; i += 1
                        self.qI.piyy[s] = qs[i]; i += 1
                        self.qI.piyn[s] = qs[i]; i += 1
                        self.qI.pinn[s] = qs[i]
                    else:
                        # store update in Q = (q + (qI + E.dt)) /2
                        qs += dt * (S + (Hx_minus - Hx_plus) / dx + (Hy_minus - Hy_plus) / dy + (Hn_minus - Hn_plus) / deta)
                        self.qI.Ttt[s]  = (self.q.Ttt[s] + qs[i]) / 2; i += 1
                        self.qI.Ttx[s]  = (self.q.Ttx[s] + qs[i]) / 2; i += 1
                        self.qI.Tty[s]  = (self.q.Tty[s] + qs[i]) / 2; i += 1
                        self.qI.Ttn[s]  = (self.q.Ttn[s] + qs[i]) / 2; i += 1
                        self.qI.pitt[s] = (self.q.pitt[s] + qs[i]) / 2; i += 1
                        self.qI.pitx[s] = (self.q.pitx[s] + qs[i]) / 2; i += 1
                        self.qI.pity[s] = (self.q.pity[s] + qs[i]) / 2; i += 1
                        self.qI.pitn[s] = (self.q.pitn[s] + qs[i]) / 2; i += 1
                        self.qI.pixx[s] = (self.q.pixx[s] + qs[i]) / 2; i += 1
                        self.qI.pixy[s] = (self.q.pixy[s] + qs[i]) / 2; i += 1
                        self.qI.pixn[s] = (self.q.pixn[s] + qs[i]) / 2; i += 1
                        self.qI.piyy[s] = (self.q.piyy[s] + qs[i]) / 2; i += 1
                        self.qI.piyn[s] = (self.q.piyn[s] + qs[i]) / 2; i += 1
                        self.qI.pinn[s] = (self.q.pinn[s] + qs[i]) / 2
        return

    # ------------------------------------------------------------------
    def recompute_euler_step(self, dt):
        '''
        This function recomputes qI after the first adaptive step
        ---------
        dt - time step size
        '''
        Nx   = self.m_lattice.lattice_points_x
        Ny   = self.m_lattice.lattice_points_y
        Neta = self.m_lattice.lattice_points_eta

        for k in range(2, Neta + 2):
            for j in range(2, Ny + 2):
                for i in range(2, Nx + 2):
                    # linear array index
                    s = self.calc_linear_array_index(i, j, k, nx + 4, ny + 4)

                    # upadte qI variabels
                    self.qI.Ttt[s]  = self.q.Ttt[s] + self.qI.Ttt[s] * dt; i += 1
                    self.qI.Ttx[s]  = self.q.Ttx[s] + self.qI.Ttx[s] * dt; i += 1
                    self.qI.Tty[s]  = self.q.Tty[s] + self.qI.Tty[s] * dt; i += 1
                    self.qI.Ttn[s]  = self.q.Ttn[s] + self.qI.Ttn[s] * dt; i += 1
                    self.qI.pitt[s] = self.q.pitt[s] + self.qI.pitt[s] * dt; i += 1
                    self.qI.pitx[s] = self.q.pitx[s] + self.qI.pitx[s] * dt; i += 1
                    self.qI.pity[s] = self.q.pity[s] + self.qI.pity[s] * dt; i += 1
                    self.qI.pitn[s] = self.q.pitn[s] + self.qI.pitn[s] * dt; i += 1
                    self.qI.pixx[s] = self.q.pixx[s] + self.qI.pixx[s] * dt; i += 1
                    self.qI.pixy[s] = self.q.pixy[s] + self.qI.pixy[s] * dt; i += 1
                    self.qI.pixn[s] = self.q.pixn[s] + self.qI.pixn[s] * dt; i += 1
                    self.qI.piyy[s] = self.q.piyy[s] + self.qI.piyy[s] * dt; i += 1
                    self.qI.piyn[s] = self.q.piyn[s] + self.qI.piyn[s] * dt; i += 1
                    self.qI.pinn[s] = self.q.pinn[s] + self.qI.pinn[s] * dt

    # ------------------------------------------------------------------
    def evolve_hydro_one_time_step(self, n, tau, dtau, dtau_prev, update, hit_CFL, model):
        '''
        This function does the full RK2 time step to update the dynamical variables
        ----------
        tau       - current proper time
        dtau      - proper time step
        dtau_prev - previous proper time step (for adaptive time step)
        update    - flag for calculate_euler_step function
        hit_CFL   - for adaptive time step, checks to see if we have hit the CFL condition
        '''
        # utility functions for this function
        def set_inferred_variables(tau):
            '''
            This function reconstructs the energy momentum tensor given the evolution 
            of the dynamic variables q
            ----------
            tau - the current proper time
            '''
            Nx   = self.m_lattice.lattice_points_x
            Ny   = self.m_lattice.lattice_points_y
            Neta = self.m_lattice.lattice_points_eta

            T_switch = self.m_hydro.freezeout_temperature_GeV
            e_switch = self.calc_conformal_energy(T_switch / hbarc)
            e_min    = self.m_hydro.energy_min

            tau2 = tau * tau
            tau4 = tau2 * tau2

            for k in range(2, Neta + 2):
                for j in range(2, Ny + 2):
                    for i in range(2, Nx + 2):
                        # linear array index
                        s = self.calc_linear_array_index(i, j, k , Nx + 4, Ny + 4)

                        # store current dynamic variables
                        Ttt = self.q.Ttt[s] 
                        Ttx = self.q.Ttx[s] 
                        Tty = self.q.Tty[s] 
                        Ttn = self.q.Ttn[s] 
                        pitt = self.q.pitt[s]
                        pitx = self.q.pitx[s]
                        pity = self.q.pity[s]
                        pitn = self.q.pitn[s]
                        pixx = self.q.pixx[s]
                        pixy = self.q.pixy[s]
                        pixn = self.q.pixn[s]
                        piyy = self.q.piyy[s]
                        piyn = self.q.piyn[s]
                        pinn = self.q.pinn[s] 

                        # construct intermediate terms
                        Mt = Ttt - pitt
                        Mx = Ttx - pitx
                        My = Tty - pity
                        Mn = Ttn - pitn
                        M  = np.sqrt(Mx ** 2 + My ** 2 + tau2 * Mn ** 2)

                        # reconstruct energy density
                        e_s = (-Mt + np.sqrt(4 *Mt ** 2 - 3 * M ** 2))
                        if e_s is np.nan:
                            sys.exit("Error: get_inferred_variables: square root of negative number occured.")
                        else:
                            # regulate energy density
                            e_cut = np.fmax(0, e_s)
                            e_s   = e_cut + e_min * np.exp(- e_cut / e_min)

                        # reconstruct fluid velocities
                        v_s  = M / (Mt + 3 * e_s)
                        ut_s = 1 / np.sqrt(1 - v_s ** 2)
                        ux_s = ut_s * Mx / (Mt + 3 * e_s)
                        uy_s = ut_s * My / (Mt + 3 * e_s)
                        un_s = ut_s * Mn / (Mt + 3 * e_s)

                        # update hydro variables
                        self.e[s]    = e_s 
                        self.u.ux[s] = ux_s
                        self.u.uy[s] = uy_s
                        self.u.un[s] = un_s

        def regulate_viscous_currents(tau):
            '''
            This function ensures that the dynamical variables stay positive where
            they are supposed to be, to avoid unphysical discontinuities.
            ---------
            tau - current proper time
            '''        
            Nx   = self.m_lattice.lattice_points_x
            Ny   = self.m_lattice.lattice_points_y
            Neta = self.m_lattice.lattice_points_eta

            tau2 = tau * tau
            tau4 = tau2 * tau2

            regulation_scheme = self.m_hydro.regulation_scheme
            for k in range(2, Neta + 2):
                for j in range(2, Ny + 2):
                    for i in range(2, Nx + 2):
                        # linear array index
                        s = self.calc_linear_array_index(i, j, k, Nx + 4, Ny + 4)

                        # calculate conformal pressure
                        P = 3 * self.e[s]

                        # store current dynamical variables
                        ux     = self.u.ux[s]
                        uy     = self.u.uy[s]
                        un     = self.u.un[s]
                        ut     = np.sqrt(1 + ux ** 2 + uy ** 2 + tau2 * un ** 2)
                        utperp = np.sqrt(1 + ux ** 2 + uy ** 2)

                        pitt = self.q.pitt[s]
                        pitx = self.q.pitx[s]
                        pity = self.q.pity[s]
                        pitn = self.q.pitn[s]
                        pixx = self.q.pixx[s]
                        pixy = self.q.pixy[s]
                        pixn = self.q.pixn[s]
                        piyy = self.q.piyy[s]
                        piyn = self.q.piyn[s]
                        pinn = self.q.pinn[s]

                        # enforce orthoganality and tracelessness conditions
                        pinn = (pixx * (ux ** 2 - ut ** 2) + piyy * (uy ** 2 - ut ** 2) + 2 * (pixy * ux * uy + tau2 * un * (pixn * ux + piyn * uy))) / (tau2 * utperp ** 2)
                        pitn = (pixn * ux + piyn * uy + tau2 * pinn * un) / ut
                        pity = (pixy * ux + piyy * uy + tau2 * piyn * un) / ut
                        pitx = (pixx * ux + pixy * uy + tau2 * pixn * un) / ut
                        pitt = (pitx * ux + pity * uy + tau2 * pitn * un) / ut

                        # equilibrium temperature?
                        Teq = np.sqrt(3) * P

                        # regulation factor
                        factor_shear = 1

                        # only regulate region wher Tmunu magnitude > equilibrium part
                        Tvisc = np.sqrt(np.fabs(pitt ** 2 + pixx ** 2 + piyy ** 2 + tau4 * pinn ** 2 - 2 * (pitx ** 2 + pity ** 2 - pixy ** 2 + tau2 * (pitn ** 2 - pixn ** 2 - piyn ** 2))))

                        #calculate regulation factor
                        factor = np.fabs(Teq / (1e-10 + Tvisc))
                        if factor < 1:
                            factor_shear = factor

                        # update shear stress
                        self.q.pitt[s] = factor_shear * pitt
                        self.q.pitx[s] = factor_shear * pitx
                        self.q.pity[s] = factor_shear * pity
                        self.q.pitn[s] = factor_shear * pitn
                        self.q.pixx[s] = factor_shear * pixx
                        self.q.pixy[s] = factor_shear * pixy
                        self.q.pixn[s] = factor_shear * pixn
                        self.q.piyy[s] = factor_shear * piyy
                        self.q.piyn[s] = factor_shear * piyn
                        self.q.pinn[s] = factor_shear * pinn


        # first intermediate step
        RK2 = 0
        if self.m_lattice.adaptive_time_step and not hit_CFL:
            if n is 0:
                self.calculate_euler_step(tau, dtau, dtau_prev, update, RK2, model)
            else:
                self.recompute_euler_step(dtau)
        else:
            self.calculate_euler_step(tau, dtau, dtau_prev, update, RK2, mdoel)

        tau += dtau

        self.swap_fluid_velocity(self.u, self.u_p)

        # reconstruct and regulate
        set_inferred_variables(tau)
        regulate_viscous_currents(tau)

        # update ghost cells
        self.set_ghost_cells()

        # second intermediate step
        RK = 1
        self.calculate_euler_step(tau, dtau, dtau_prev, update, RK2, model)
        set_inferred_variables(tau)
        regulate_viscous_currents(tau)

        # one last swap
        self.swap_hydro_variables(self.q, self.Q)
        self.set_ghost_cells()
        return
    
    # ------------------------------------------------------------------
    def set_time_step(self, n, tau, dtau_prev, hit_CFL_bound):
        '''
        Function used to calculate adaptive time step
        ---------
        tau           - current proper time
        dtau_prev     - the previous time step 
        t_next_output - tau + dtau_prev (generally, but could be something else)
        ---------
        Returns:
        dtau - the updated time step
        '''
        # utility functions
        def compute_dt_CFL(tau):
            '''
            Compute CFL time step for grid we are solving on
            ---------
            tau - current proper time
            '''
            # starting point
            dt_CFL = np.inf
            
            Nx   = self.m_lattice.lattice_points_x
            Ny   = self.m_lattice.lattice_points_y
            Neta = self.m_lattice.lattice_points_eta
            
            dx   = self.m_lattice.lattice_spacing_x
            dy   = self.m_lattice.lattice_spacing_y
            deta = self.m_lattice.lattice_spacing_eta
            
            tau2 = tau * tau
            theta = self.m_hydro.flux_limiter
            
            stride_y = Nx + 4
            stride_eta = (Nx + 4) * (Ny + 4)
            for k in range(2, Neta + 2):
                for j in range(2, Ny + 2):
                    for i in range(2, Nx + 2):
                        # for get_fluid_velocity_neighbors function
                        ui1 = np.zeros(6)
                        uj1 = np.zeros(6)
                        uk1 = np.zeros(6)
                        # these contain the 1st and 2nd order neighbors
                        vxi = np.zeros(4)    # {i-2,i-1,i+1,i+2}
                        vyj = np.zeros(4)    # {j-2,j-1,j+1,j+2}
                        vnk = np.zeros(4)    # {k-2,k-1,k+1,k+2}
                        
                        # linear array index
                        s = calc_linear_array_index(i, j, k, Nx + 4, Ny + 4)
                        
                        # neighboring x-indices
                        simm = s - 2
                        sim  = s - 1
                        sip  = s + 1
                        sipp = s + 2
                        # neighboring y-indices
                        sjmm = s - 2 * stride_y
                        sjm  = s - stride_y
                        sjp  = s + stride_y
                        sjpp = s + 2 * stride_y
                        # neighboring eta-indices
                        skmm = s - 2 * stride_eta
                        skm  = s - stride_eta
                        skp  = s + stride_eta
                        skpp = s + 2 * stride_eta
                        
                        # store current fluid velocities
                        ux = self.u.ux[s]
                        uy = self.u.uy[s]
                        un = self.u.un[s]
                        ut = np.sqrt(1 + ux ** 2 + uy ** 2 + tau2 * un ** 2)
                        
                        ui1, uj1, uk1, vxi, vyj, vnk = self.get_fluid_velocity_neighbors(simm, sim, sip, sipp,
                                                                                sjmm, sjm, sjp, sjpp,
                                                                                skmm, skm, skp, skpp,
                                                                                tau2)
                        # locao propogation speeds
                        ax = self.compute_max_local_propogation_speed(vxi, ux / ut)
                        ay = self.compute_max_local_propogation_speed(vyj, uy / ut)
                        an = self.compute_max_local_propogation_speed(vnk, un / ut)
                        # take the smallest one
                        dt_CFL = np.fmin(dt_CFL, np.fmin(dx / ax, np.fmin(dy / ay, deta / an)))
            return dt_CFL / 8
            
        def compute_dt_source(self, tau, dt_prev):
            '''
            Computes time step resolution given the hydrodynamic update variables
            ---------
            tau - current proper time
            '''
            # utility functions for computing dt_source
            def compute_q_star(s, dt_prev):
                '''
                Creates a hydro_variable instance containing an intermediate calculating result
                ---------
                s       - linear array index hydro variables
                dt_prev - previous time step
                '''
                q_star = self.hydro_variables(1)
                # set q_star values
                q_star.Ttt  = self.q.Ttt[s] + dt_prev * self.qI.Ttt[s]
                q_star.Ttx  = self.q.Ttx[s] + dt_prev * self.qI.Ttx[s]
                q_star.Tty  = self.q.Tty[s] + dt_prev * self.qI.Tty[s]
                q_star.Ttn  = self.q.Ttn[s] + dt_prev * self.qI.Ttn[s]
                q_star.pitt = self.q.pitt[s] + dt_prev * self.qI.pitt[s]
                q_star.pitx = self.q.pitx[s] + dt_prev * self.qI.pitx[s]
                q_star.pity = self.q.pity[s] + dt_prev * self.qI.pity[s]
                q_star.pitn = self.q.pitn[s] + dt_prev * self.qI.pitn[s]
                q_star.pixx = self.q.pixx[s] + dt_prev * self.qI.pixx[s]
                q_star.pixy = self.q.pixy[s] + dt_prev * self.qI.pixy[s]
                q_star.pixn = self.q.pixn[s] + dt_prev * self.qI.pixn[s]
                q_star.piyy = self.q.piyy[s] + dt_prev * self.qI.piyy[s]
                q_star.piyn = self.q.piyn[s] + dt_prev * self.qI.piyn[s]
                q_star.pinn = self.q.pinn[s] + dt_prev * self.qI.pinn[s]
                return q_star
            
            def compute_hydro_norm2(s, tau, flag):
                '''
                Sums up all the dynamics vairables.
                <Might want to check how adding same united objects affects this>
                '''
                if flag is 1: # compute for q
                    norm2  = self.q.Ttt[s] ** 2 + self.q.Ttx[s] ** 2 + self.q.Tty ** 2 + self.q.Ttn[s]
                    norm2 += self.q.pitt[s] ** 2 + self.q.pitx[s] ** 2 + self.q.pity[s] ** 2 + self.q.pitn[s] ** 2
                    norm2 += self.q.pixx[s] ** 2 + self.q.pixy[s] ** 2 + self.q.piyy[s] ** 2
                    norm2 += self.q.pixn[s] ** 2 + self.q.piyn[s] ** 2
                    norm2 += self.q.pinn[s]
                elif flag is 2: # compute for Q
                    norm2  = self.Q.Ttt[s] ** 2 + self.Q.Ttx[s] ** 2 + self.Q.Tty ** 2 + self.Q.Ttn[s]
                    norm2 += self.Q.pitt[s] ** 2 + self.Q.pitx[s] ** 2 + self.Q.pity[s] ** 2 + self.Q.pitn[s] ** 2
                    norm2 += self.Q.pixx[s] ** 2 + self.Q.pixy[s] ** 2 + self.Q.piyy[s] ** 2
                    norm2 += self.Q.pixn[s] ** 2 + self.Q.piyn[s] ** 2
                    norm2 += self.Q.pinn[s]
                return norm2
            
            def compute_dot_product(s, tau):
                # worried about units hear as well.
                dot_product  = self.q.Ttt[s] * self.qI.Ttt[s]
                dot_product += self.q.Ttx[s] * self.qI.Ttx[s]
                dot_product += self.q.Tty[s] * self.qI.Tty[s]
                dot_product += self.q.Ttn[s] * self.qI.Ttn[s]
                dot_product += self.q.pitt[s] * self.qI.pitt[s]
                dot_product += self.q.pitx[s] * self.qI.pitx[s]
                dot_product += self.q.pity[s] * self.qI.pity[s]
                dot_product += self.q.pitn[s] * self.qI.pitn[s]
                dot_product += self.q.pixx[s] * self.qI.pixx[s]
                dot_product += self.q.pixy[s] * self.qI.pixy[s]
                dot_product += self.q.pixn[s] * self.qI.pixn[s]
                dot_product += self.q.piyy[s] * self.qI.piyy[s]
                dot_product += self.q.piyn[s] * self.qI.piyn[s]
                dot_product += self.q.pinn[s] * self.qI.pinn[s]
                return dot_product
            
            def compute_second_derivative_squared(q_prev, q, q_star):
                return (q_prev - 2 * q + q_star) ** 2
            
            def compute_second_derivative_norm(s, q_star):
                # units again
                norm2  = compute_second_derivative_squared(self.Q.Ttt[s], self.q.Ttt[s], q_star.Ttt)
                norm2 += compute_second_derivative_squared(self.Q.Ttx[s], self.q.Ttx[s], q_star.Ttx)
                norm2 += compute_second_derivative_squared(self.Q.Tty[s], self.q.Tty[s], q_star.Tty)
                norm2 += compute_second_derivative_squared(self.Q.Ttn[s], self.q.Ttn[s], q_star.Ttn)
                norm2 += compute_second_derivative_squared(self.Q.pitt[s], self.q.pitt[s], q_star.pitt)
                norm2 += compute_second_derivative_squared(self.Q.pitx[s], self.q.pitx[s], q_star.pitx)
                norm2 += compute_second_derivative_squared(self.Q.pity[s], self.q.pity[s], q_star.pity)
                norm2 += compute_second_derivative_squared(self.Q.pitn[s], self.q.pitn[s], q_star.pitn)
                norm2 += compute_second_derivative_squared(self.Q.pixx[s], self.q.pixx[s], q_star.pixx)
                norm2 += compute_second_derivative_squared(self.Q.pixy[s], self.q.pixy[s], q_star.pixy)
                norm2 += compute_second_derivative_squared(self.Q.pixn[s], self.q.pixn[s], q_star.pixn)
                norm2 += compute_second_derivative_squared(self.Q.piyy[s], self.q.piyy[s], q_star.piyy)
                norm2 += compute_second_derivative_squared(self.Q.piyn[s], self.q.piyn[s], q_star.piyn)
                norm2 += compute_second_derivative_squared(self.Q.pinn[s], self.q.pinn[s], q_star.pinn)
                return np.sqrt(norm2)
            
            def adaptive_method_norm(q_norm2, f_norm2, second_deriv_norm, q_dot_f, dt_prev, delta_0):
                sqrt_vars = np.sqrt(self.m_hydro.dynamical_variables)
                dt_abs  = dt_prev * np.sqrt(delta_0 * sqrt_vars / second_deriv_norm)
                dt_abs2 = dt_abs * dt_abs / sqrt_vars
                dt_abs4 = dt_abs2 * dt_abs2
                
                # solve quartic equation x^4 - c.x^2 - b.x - a = 0 (x = dt_rel) 
                c = f_norm2 * dt_abs4
                b = 2 * q_dot_f * dt_abs4
                a = q_norm2 * dt_abs4
                roots = np.roots([1, 0, -c, -b, -a])
                if np.max(roots) > 0 and not np.iscomplex(np.max(roots)):
                    return np.max(roots)
                else:
                    return dt_abs
            # begin function implementation - compute_dt_source
            # initialize to infinity
            dt_source = np.inf
            
            Nx   = self.m_lattice.lattice_points_x
            Ny   = self.m_lattice.lattice_points_y
            Neta = self.m_lattice.lattice_points_eta
            
            alpha   = self.m_hydro.alpha
            delta_0 = self.m_hydro.delta_0
            for k in range(2, Neta + 2):
                for j in range(2, Ny + 2):
                    for i in range(2, Nx + 2):
                        # linear array index
                        s = calc_linear_array_index(i, j, k, Nx + 4, Ny + 4)
                        
                        # intermediate variables
                        q_star  = compute_q_star(s, dt_prev)
                        q_norm2 = compute_hydro_norm2(s, tau, 1)
                        f_norm2 = compute_hydro_norm2(s, tau, 2)
                        q_dot_f = dot_product(s, tau)
                        
                        second_deriv_norm = compute_second_derivative_norm(s, q_star)
                        dt_source = np.fmin(dt_source, adaptive_method_norm(q_norm2, f_norm2, second_deriv_norm, q_dot_f,  dt_prev, delta_0))
            dt_source = np.fmax((1 - alpha) * dt_prev, np.fmin(dt_source, (1 + alpha) * dt_prev))
            return dt_source
        
        def compute_adaptive_time_step(tau, dt_CFL, dt_source, dt_min):
            dt = np.fmin(dt_CFL, dt_source)
            # round dt to numerical precision
            dt = 0.001 * dt_min * np.floor(1000 * dt / dt_min) 
            dt = np.fmax(dt_min, dt)
            return dt
        # begin function implementain : set_time_step
        dt_min = self.m_hydro.tau_initial / 20
        dt     = self.m_lattice.fixed_time_step
        
        if (self.m_lattice.adaptive_time_step):
            if n is 0:
                dt = dt_min
            else:
                dt_CFL = compute_dt_CFL(tau)
                dt_source = np.inf
                if not hit_CFL_bound:
                    update = 0
                    self.calculate_euler_step(tau, dt_prev, dt_prev, update, hit_CFL_bound)
                    
                    dt_source = compute_dt_source(tau, dt_prev)
                    if dt_source > dt_CFL:
                        print(f'\nHit CFL bound at t = {tau}')
                        hit_CFL_bound = True
                dt = compute_adaptive_time_step(tau, dt_CFL, dt_source, dt_min)
        return dt, hit_CFL_bound
                        
    # ------------------------------------------------------------------    
    def run_hydro(self, model):
        '''
        This function is called to run the hydrodynamic simulation. 
        It takes as input, a class Hydro_Model which perscribes the equations 
        of motion to be solved.
        ----------
        model - instance of Hydro_Model class. Specifies sources terms for KT calculation and more
        '''
        print('Begin simulation.')
        # allocate memory for dynamical variables
        print('Allocating memory.')
        self.setup_lattice()
        self.allocate_dynamical_variable()
        
        # starting proper time and time step
        tau  = self.m_hydro.tau_initial
        dtau = self.m_hydro.tau_initial / 20
        
        if not self.m_lattice.adaptive_time_step:
            dtau = self.m_lattice.fixed_time_step
        
        dtau_p = dtau
        
        # setup boundary conditions
        print('Setting inital conditions.')
        self.set_BC(model=model, tau=tau)
        self.set_ghost_cells()
        
        # simulation loop
        hit_CFL_bound = False
        for n in range(self.m_lattice.max_time_steps):
            print(f'Time step: {n}')
            if not self.check_all_cells_below_freezeout():
                # set time step and check CFL condition
                dt, hit_CFL_bound = self.set_time_step(n, tau, dtau_p, hit_CFL_bound)
                
                # evolve system by one time step
                update = 1
                self.evolve_hydro_one_time_step(n, tau, dtau, dtau_p, update, hit_CFL_bound, model)
                tau += dtau
                dtau_p = dtau
            else:
                break
        print('Simulation has completed.\n')