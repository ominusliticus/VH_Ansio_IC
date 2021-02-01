import numpy as np

# global constant
hbarc = 0.197326938 

# this class will house the details of the model we want to solve 
# with th Hydro_Sim class above
class Hydro_Model:
    def __init__(self, model_flag, eta_s):
        '''
        Constructor - takes in model flag which specifies which hydrodynamic 
        model to use.
        ---------
        model_flag == 1 - MIS full
        model_flag == 2 - MIS linear fluctuations
        '''
        self.m_model_flag = model_flag
        self.m_eta_s      = eta_s
    
    # ------------------------------------------------------------------
    class transport_coefficients:
        def __init__(self, model_flag, T_in, e_in, p_in):
            '''
            Constructor for the tansport coefficients. 
            ---------
            model_flag == 1 - MIS full
            model_flag == 2 - MIS linear fluctuation
            ---
            T_in            - input temperature
            '''
            self.m_T = T_in
            self.m_e = e_in
            if model_flag is 1:
                self.m_p = p_in
                
        def compute_transport_coefficients(self, model_flag, eta_s):
            '''
            Calculates and returns the transport coefficients for 
            various hydrodynamic models
            ------
            model_flag == 1 - MIS full
            model_flag == 2 - MIS linear fluctuation
            '''
            if model_flag is 1: # MIS full
                # need to edit when ready
                tau_pi  = eta_s / self.m_T
                beta_pi = (self.m_p + self.m_e) / 5
                return tau_pi, beta_pi
            elif model_flag is 2: # MIS linear fluctuation
                # this will come later  
                print('Option not yet available')
            else:
                print('Invalid option')
    
    class spatial_projection:
        def __init__(self, Dtt, Dtx, Dty, Dtn, Dxx, Dxy, Dxn, Dyy, Dyn, Dnn):
            '''
            Constructor for spatial projection tensor
            --------
            D     - elements of a spatial projection tensor
            '''
            self.Dtt = Dtt
            self.Dtx = Dtx
            self.Dty = Dty
            self.Dtn = Dtn
            self.Dxx = Dxx
            self.Dxy = Dxy
            self.Dxn = Dxn
            self.Dyy = Dyy
            self.Dyn = Dyn
            self.Dnn = Dnn
        
    def project_spatial(self, Delta, At, Ax, Ay, An):
        '''
        Caluclates the projecting for a 4-vector A
        -------
        Delta - instance of spatial_projection class
        A     - 4-vector components
        '''
        At_proj = Delta.Dtt * At - Delta.Dtx * Ax - Delta.Dty * Ay - Delta.Dtn * An
        Ax_proj = Delta.Dtx * At - Delta.Dxx * Ax - Delta.Dxy * Ay - Delta.Dxn * An
        Ay_proj = Delta.Dty * At - Delta.Dxy * Ax - Delta.Dyy * Ay - Delta.Dyn * An
        An_proj = Delta.Dtn * At - Delta.Dxn * Ax - Delta.Dyn * Ay - Delta.Dnn * An
        return At_proj, Ax_proj, Ay_pron, An_proj
            
    class symmetric_traceless_projection:
        def __init__(self, Dtt, Dtx, Dty, Dtn, Dxx, Dxy, Dxn, Dyy, Dyn, Dnn, t2):
            '''
            Constructor for symmetric traceless tensors
            ------
            D     - elements of a spatial projection tensor
            t2    - proper time squared
            '''
            self.t2 = t2
            self.t4 = t2 * t2
            
            self.Dtt_tt = 2 / 3 * Dtt * Dtt
            self.Dtt_tx = 2 / 3 * Dtt * Dtx
            self.Dtt_ty = 2 / 3 * Dtt * Dty
            self.Dtt_tn = 2 / 3 * Dtt * Dtn
            self.Dtt_xx = Dtx * Dtx - Dtt * Dxx / 3
            self.Dtt_xy = Dtx * Dty - Dtt * Dxy / 3
            self.Dtt_xn = Dtx * Dtn - Dtt * Dxn / 3
            self.Dtt_yy = Dty * Dty - Dtt * Dyy / 3
            self.Dtt_yn = Dty * Dtn - Dtt * Dyn / 3
            self.Dtt_nn = Dtn * Dtn - Dtt * Dnn / 3
            
            self.Dtx_tx = (Dtx * Dtx + 3 * Dtt * Dxx) / 6
            self.Dtx_ty = (Dtx * Dty + 3 * Dtt * Dxy) / 6
            self.Dtx_tn = (Dtx * Dtn + 3 * Dtt * Dxn) / 6
            self.Dtx_xx = 2 / 3 * Dtx * Dxx
            self.Dtx_xy = (Dtx * Dxy + 3 * Dty * Dxx) / 6
            self.Dtx_xn = (Dtx * Dxn + 3 * Dtn * Dxx) / 6
            self.Dtx_yy = Dty * Dxy - Dtx * Dyy / 3
            self.Dtx_yn = (3 * Dty * Dxn + 3 * Dtn * Dxy - 2 * Dtx * Dyn) / 6
            self.Dtx_nn = Dtn * Dxn - Dtx * Dnn / 3
            
            self.Dty_ty = (Dty * Dty + 3 * Dtt * Dyy) / 6
            self.Dty_tn = (Dty * Dtn + 3 * Dtt * Dyn) / 6
            self.Dty_xx = Dtx * Dxy - Dty * Dxx / 3
            self.Dty_xy = (Dty * Dxy + 3 * Dtx * Dyy) / 6
            self.Dty_xn = (3 * Dtx * Dyn + 3 * Dtn * Dxy - 2 * Dty * Dxn) / 6
            self.Dty_yy = 2 / 3 * Dty * Dyy
            self.Dty_yn = (Dty * Dyn + 3 * Dtn * Dyy) / 6
            self.Dty_nn = Dtn * Dyn - Dnn * Dty / 3
            
            self.Dtn_tn = (Dtn * Dtn + 3 * Dnn * Dtt) / 6
            self.Dtn_xx = Dtx * Dxn - Dtn * Dxx / 3
            self.Dtn_xy = (3 * Dty * Dxn + 3 * Dtx * Dyn - 2 * Dtn * Dxy) / 6
            self.Dtn_xn = (Dtn * Dxn + 3 * Dnn * Dtx) / 6
            self.Dtn_yy = Dty * Dyn - Dtn * Dyy / 3
            self.Dtn_yn = (Dtn * Dyn  + 3 * Dnn * Dty)
            self.Dtn_nn = 2 / 3 * Dnn * Dtn
            
            self.Dxx_xx = 2 / 3 * Dxx * Dxx
            self.Dxx_xy = 2 / 3 * Dxx * Dxy
            self.Dxx_xn = 2 / 3 * Dxx * Dxn
            self.Dxx_yy = Dxy * Dxy - Dxx * Dyy / 3
            self.Dxx_yn = Dxy * Dxn - Dxx * Dyn / 3
            self.Dxx_nn = Dxn * Dxn - Dxx * Dnn / 3
            
            self.Dxy_xy = (Dxy * Dxy + 3 * Dxx * Dyy) / 6
            self.Dxy_xn = (Dxn * Dxy + 3 * Dxx * Dyn) / 6
            self.Dxy_yy = 2 / 3 * Dyy * Dxy
            self.Dxy_yn = (Dxy * Dyn + 3 * Dxn * Dxy) / 6
            self.Dxy_nn = Dxn * Dyn - Dxy * Dnn / 3
            
            self.Dxn_xn = (Dxn * Dxn + 3 * Dnn * Dxx) / 6
            self.Dxn_yy = Dxy * Dyn - Dyy * Dxn / 3
            self.Dxn_yn = (Dxn * Dyn + 3 * Dnn * Dxy) / 6
            self.Dxn_nn = Dxn * Dyn - Dxy * Dnn / 3
            
            self.Dyy_yy = 2 / 3 * Dyy * Dyy
            self.Dyy_yn = 2 / 3 * Dyy * Dyn
            self.Dyy_nn = Dyn * Dyn - Dyy * Dnn / 3
            
            self.Dyn_yn = (Dyn * Dyn + 3 * Dnn * Dyy) / 6
            self.Dyn_nn = 2 / 3 * Dyn * Dnn
            
            self.Dnn_nn = 2 / 3 * Dnn * Dnn   
            
        
    def project_sym_tr(self, Delta, Att, Atx, Aty, Atn, Axx, Axy, Axn, Ayy, Ayn, Ann):
        '''
        Project out the symmetric traceless part of a two-index tensor A
        -------
        Delta - instance of symmetric_traceless_projection class
        A     - components of a symmetric 2-tensor
        '''
        Att_proj = Delta.Dtt_tt * Att + Delta.Dtt_xx * Axx + Delta.Dtt_yy * Ayy + Delta.t4 * Delta.Dtt_nn * Ann - 2 * (Delta.Dtt_tx * Atx + Delta.Dtt_ty * Aty - Delta.Dtt_xy * Axy + Delta.t2 * (Delta.Dtt_tn * Atn - Delta.Dtt_xn * Axn - Delta.Dtt_yn * Ayn))
        Atx_proj = Delta.Dtt_tx * Att + Delta.Dtx_xx * Axx + Delta.Dtx_yy * Ayy + Delta.t4 * Delta.Dtx_nn * Ann - 2 * (Delta.Dtx_tx * Atx + Delta.Dtx_ty * Aty - Delta.Dtx_xy * Axy + Delta.t2 * (Delta.Dtx_tn * Atn - Delta.Dtx_xn * Axn - Delta.Dtx_yn * Ayn))
        Aty_proj = Delta.Dtt_ty * Att + Delta.Dty_xx * Axx + Delta.Dty_yy * Ayy + Delta.t4 * Delta.Dty_nn * Ann - 2 * (Delta.Dtx_ty * Atx + Delta.Dty_ty * Aty - Delta.Dty_xy * Axy + Delta.t2 * (Delta.Dty_tn * Atn - Delta.Dty_xn * Axn - Delta.Dty_yn * Ayn))
        Atn_proj = Delta.Dtt_tn * Att + Delta.Dtn_xx * Axx + Delta.Dtn_yy * Ayy + Delta.t4 * Delta.Dtn_nn * Ann - 2 * (Delta.Dtx_tn * Atx + Delta.Dty_tn * Aty - Delta.Dtn_xy * Axy + Delta.t2 * (Delta.Dtn_tn * Atn - Delta.Dtn_xn * Axn - Delta.Dtn_yn * Ayn))
        Axx_proj = Delta.Dtt_xx * Att + Delta.Dxx_xx * Axx + Delta.Dxx_yy * Ayy + Delta.t4 * Delta.Dxx_nn * Ann - 2 * (Delta.Dtx_xx * Atx + Delta.Dty_xx * Aty - Delta.Dxx_xy * Axy + Delta.t2 * (Delta.Dtn_xx * Atn - Delta.Dxx_xn * Axn - Delta.Dxx_yn * Ayn))
        Axy_proj = Delta.Dtt_xy * Att + Delta.Dxx_xy * Axx + Delta.Dxy_yy * Ayy + Delta.t4 * Delta.Dxy_nn * Ann - 2 * (Delta.Dtx_xy * Atx + Delta.Dty_xy * Aty - Delta.Dxy_xy * Axy + Delta.t2 * (Delta.Dtx_xy * Atn - Delta.Dxy_xn * Axn - Delta.Dxy_yn * Ayn))
        Axn_proj = Delta.Dtt_xn * Att + Delta.Dtx_xn * Axx + Delta.Dxn_yy * Ayy + Delta.t4 * Delta.Dxn_nn * Ann - 2 * (Delta.Dtx_xn * Atx + Delta.Dty_xn * Aty - Delta.Dxy_xn * Axy + Delta.t2 * (Delta.Dtn_xn * Atn - Delta.Dxn_xn * Axn - Delta.Dxn_yn * Ayn))
        Ayy_proj = Delta.Dtt_yy * Att + Delta.Dxx_yy * Axx + Delta.Dyy_yy * Ayy + Delta.t4 * Delta.Dyy_nn * Ann - 2 * (Delta.Dtx_yy * Atx + Delta.Dty_yy * Aty - Delta.Dxy_yy * Axy + Delta.t2 * (Delta.Dtx_yy * Atn - Delta.Dxn_yy * Axn - Delta.Dyy_yn * Ayn))
        Ayn_proj = Delta.Dtt_yn * Att + Delta.Dxx_yn * Axx + Delta.Dyy_yn * Ayy + Delta.t4 * Delta.Dyn_nn * Ann - 2 * (Delta.Dtx_yn * Atx + Delta.Dty_yn * Aty - Delta.Dxy_yn * Axy + Delta.t2 * (Delta.Dtn_yn * Atn - Delta.Dxn_yn * Axn - Delta.Dyn_yn * Ayn))
        Ann_proj = Delta.Dtt_nn * Att + Delta.Dxx_nn * Axx + Delta.Dyy_nn * Ayy + Delta.t4 * Delta.Dnn_nn * Ann - 2 * (Delta.Dtx_nn * Atx + Delta.Dty_nn * Aty - Delta.Dxy_nn * Axy + Delta.t2 * (Delta.Dtn_nn * Atn - Delta.Dxn_nn * Axn - Delta.Dyn_nn * Ayn))
        return Att_proj, Atx_proj, Aty_proj, Atn_proj, Axx_proj, Axy_proj, Axn_proj, Ayy_proj, Ayn_proj, Ann_proj
    
    # ------------------------------------------------------------------     
    def conformal_factor(self):
        colors = 3.
        flavors = 3.
        return np.pi ** 2 * (2 * (colors ** 2 - 1) + 3.5 * colors * flavors) / 30.
    
    # ------------------------------------------------------------------     
    def background_evolution(self, veps_0, pi_0, t_0, t):
        '''
        This is the function that describes the evolution of the background fields
        It will include many different models, but for the founding of this 
        program, we only include the MIS theory:
        -----------------------------
        1: MIS - full
        2: MIS - linear fluctuations
        -----------------------------
        Need to check for consistency of units
        '''
        # speed of sound 
        cs2 = 1. / 3.
        
        # begin model specific implementations
        if self.m_model_flag is 1:
            # just return the values given
            # TODO: need to figure out what natural conditions are for pi_0
            return veps_0, pi_0
        
        elif self.m_model_flag is 2:
            fixed_dt = g_fixed_time_step
            steps = int((t - t_0) / fixed_dt)
            tau = np.linspace(t_0, t, steps, endpoint=True) # GeV^-1
            dtau = tau[1] - tau[0]
            
            # functions describing time evolution for background
            def dveps_dtau(tau, veps, pi):
                return - ((1 + cs2) * veps - pi) / tau
            
            def dpi_dtau(tau, veps, pi):
                g = (2 * (3 ** 2 - 1) + 3.5 * 3 * 3)
                T = (30 * veps / np.pi ** 2 / g) ** (0.25)
                tau_pi = self.m_eta_bar / (T / hbarc) 
                return  - (1 / tau_pi + (1 + cs2) / tau) * pi + 16 * cs2 * veps / (15 * tau)
            
            # initializing variables
            veps_cur = veps_0
            veps_next = 0.
            
            pi_cur = pi_0
            pi_next = 0.
            
            # calculate RK4 
            for i in range(1, steps):
                k1 = dveps_dtau(tau[i-1], veps_cur, pi_cur)
                l1 = dpi_dtau(tau[i-1], veps_cur, pi_cur)
                
                k2 = dveps_dtau(tau[i-1] + dtau / 2, veps_cur + k1 * dtau /2, pi_cur + l1 * dtau / 2)
                l2 = dpi_dtau(tau[i-1] + dtau / 2, veps_cur + k1 * dtau /2, pi_cur + l1 * dtau / 2)
                
                k3 = dveps_dtau(tau[i-1] + dtau / 2, veps_cur + k2 * dtau /2, pi_cur + l2 * dtau / 2)
                l3 = dpi_dtau(tau[i-1] + dtau / 2, veps_cur + k2 * dtau /2, pi_cur + l2 * dtau / 2)
                
                k4 = dveps_dtau(tau[i-1] + dtau, veps_cur + k3 * dtau, pi_cur + l3 * dtau)
                l4 = dpi_dtau(tau[i-1] + dtau, veps_cur + k3 * dtau, pi_cur + l3 * dtau)
                
                veps_next = veps_cur + (k1 + 2*k2 + 2*k3 + k4) * dtau / 6 
                v_next = pi_cur + (l1 + 2*l2 + 2*l3 + l4) * dtau / 6
                
                veps_cur = veps_next
                pi_cur   = pi_next
                
            return veps_cur, pi_cur
        
    # ------------------------------------------------------------------
    def source_ternms(self, q, e, t, qi1, qj1, qk1, e1, ui1, uj1, uk1, ux, uy, un, ux_p, uy_p, un_p, dt_prev, dx, dy, deta, hydro):
        '''
        This is the most important input function for the KT method. 
        --------
        q    - current values for dynamical variables -> np.ndarray
        e    - current eneryg density -> np.ndarray
        t    - current proper time
        ux   - current x-direction fluid velocity     -> np.ndarray
        uy   - current y-direction fluid velocity     -> np.ndarray
        un_p - previous eta-direction fluid velocity
        ux_p - previous x-direction fluid velocity
        uy_p - previous y-direction fluid velocity
        un_p - previous eta-direction fluid velocity

        qi1 - [i-1,i+1] neighbors for the dynamical variables x-coordinate -> np.ndarray
        qj1 - [j-1,j+1] neighbors for the ''        ''        y-coordinate -> np.ndarray
        qk1 - [k-1,k+1] ''        ''  ''  ''        ''        z-coordinate -> np.ndarray
        ui1 - [i-1,i+1] neighbors for the fluid velocity x-coordinate      -> np.ndarray 
        uj1 - [j-1,j+1] neighbors for the ''    ''       y-coordinate      -> np.ndarray
        uk1 - [k-1,k+1] ''        ''  ''  ''    ''       z-coordinate      -> np.ndarray
        e1  - first order neighbors for current energy density             -> np.ndarray

        dt_prev - previous time step
        dx      - x lattice spacing
        dy      - y lattice spacing
        deta    - eta lattice spacing 
        hydro   - instance of hydro_parameters subclass of Hydro_Sim class
        '''
        if self.m_model_flag is 1:
            # utility functions
            def central_difference(f, n, dx):
                return (f[n + 1] - f[n]) / (2 * dx)

            def call_spatial_projection(ut, ux, uy, un, t2):
                '''
                Calculates the spatial projection of a one index tensor
                In literature, this is often referred to as the Delta^{\mu\nu} tensor
                '''
                Dtt = 1 - ut * ut
                Dtx = -ut * ux
                Dty = -ut * uy
                Dtn = -ut * un
                Dxx = 1 - ux * ux
                Dxy = -ux * uy
                Dxn = -ux * un
                Dyy = 1 - uy * uy
                Dyn = -uy * un
                Dnn = 1 / t2 - un * un

                Delta = self.spatial_projection(Dtt, Dtx, Dty, Dtn, Dxx, Dxy, Dxn, Dyy, Dyn, Dnn)
                return Delta

            def call_symmetric_traceless_projection(ut, ux, uy, un, t2):
                '''
                Calculates the spatial projection of a one index tensor
                In literature, this is often referred to as the Delta^{\mu\nu} tensor
                '''
                Dtt = 1 - ut * ut
                Dtx = -ut * ux
                Dty = -ut * uy
                Dtn = -ut * un
                Dxx = 1 - ux * ux
                Dxy = -ux * uy
                Dxn = -ux * un
                Dyy = 1 - uy * uy
                Dyn = -uy * un
                Dnn = 1 / t2 - un * un

                Delta = self.symmetric_traceless_projection(Dtt, Dtx, Dty, Dtn, Dxx, Dxy, Dxn, Dyy, Dyn, Dnn, t2)
                return Delta

            # set up some variables for computation
            t2 = t * t 
            t4 = t2 * t2
            tun = t * un

            ut = np.sqrt(1 + ux * ux + uy * uy + tun * tun)
            vx = ux / ut
            vy = uy / ut
            vn = un / ut

            # equation of state vars
            cs2 = 1 / 3
            p   = cs2 * e
            T   = hbarc * np.power(e / self.conformal_factor(), 0.25)

            # dynamical vars
            n = 0
            Ttt = q[n]; n += 1
            Ttx = q[n]; n += 1
            Tty = q[n]; n += 1
            Ttn = q[n]; n += 1
            pitt = q[n]; n += 1
            pitx = q[n]; n += 1
            pity = q[n]; n += 1
            pitn = q[n]; n += 1
            pixx = q[n]; n += 1
            pixy = q[n]; n += 1
            pixn = q[n]; n += 1
            piyy = q[n]; n += 1
            piyn = q[n]; n += 1
            pinn = q[n]; n += 1

            # transport coefficients
            viscous = self.transport_coefficients(self.m_model_flag, T, e, p)
            tau_pi, beta_pi = viscous.compute_transport_coefficients(self.m_model_flag, self.m_eta_s)

            # calculate derivatives
            de_dx = central_difference(e1, 0, dx)
            de_dy = central_difference(e1, 2, dy)
            de_dn = central_difference(e1, 4, deta)

            dp_dx = cs2 * de_dx
            dp_dy = cs2 * de_dy
            dp_dn = cs2 * de_dn

            n = 8 # index at which pitt neighbors are
            dpitt_dx = central_difference(qi1, n, dx)
            dpitt_dy = central_difference(qj1, n, dy)
            dpitt_dn = central_difference(qk1, n, deta); n += 2

            dpitx_dx = central_difference(qi1, n, dx)
            dpitx_dy = central_difference(qj1, n, dy)
            dpitx_dn = central_difference(qk1, n, deta); n += 2

            dpity_dx = central_difference(qi1, n, dx)
            dpity_dy = central_difference(qj1, n, dy)
            dpity_dn = central_difference(qk1, n, deta); n += 2

            dpitn_dx = central_difference(qi1, n, dx)
            dpitn_dy = central_difference(qj1, n, dy)
            dpitn_dn = central_difference(qk1, n, deta); n += 2

            dpixx_dx = central_difference(qi1, n, dx)
            dpixx_dy = central_difference(qj1, n, dy)
            dpixx_dn = central_difference(qk1, n, deta); n += 2

            dpixy_dx = central_difference(qi1, n, dx)
            dpixy_dy = central_difference(qj1, n, dy)
            dpixy_dn = central_difference(qk1, n, deta); n += 2

            dpixn_dx = central_difference(qi1, n, dx)
            dpixn_dy = central_difference(qj1, n, dy)
            dpixn_dn = central_difference(qk1, n, deta); n += 2

            dpiyy_dx = central_difference(qi1, n, dx)
            dpiyy_dy = central_difference(qj1, n, dy)
            dpiyy_dn = central_difference(qk1, n, deta); n += 2

            dpiyn_dx = central_difference(qi1, n, dx)
            dpiyn_dy = central_difference(qj1, n, dy)
            dpiyn_dn = central_difference(qk1, n, deta); n += 2

            dpinn_dx = central_difference(qi1, n, dx)
            dpinn_dy = central_difference(qj1, n, dy)
            dpinn_dn = central_difference(qk1, n, deta)

            n = 0
            dux_dt = (ux - ux_p) / dt_prev
            dux_dx = central_difference(ui1, n, dx)
            dux_dy = central_difference(uj1, n, dy)
            dux_dn = central_difference(uk1, n, deta); n += 2

            duy_dt = (ux - ux_p) / dt_prev
            duy_dx = central_difference(ui1, n, dx)
            duy_dy = central_difference(uj1, n, dy)
            duy_dn = central_difference(uk1, n, deta); n += 2

            dun_dt = (ux - ux_p) / dt_prev
            dun_dx = central_difference(ui1, n, dx)
            dun_dy = central_difference(uj1, n, dy)
            dun_dn = central_difference(uk1, n, deta)

            dut_dt = vx * dux_dt + vy * duy_dt + t2 * vn * dun_dt + tun * vn
            dut_dx = vx * dux_dx + vy * duy_dx + t2 * vn * dun_dx 
            dut_dy = vx * dux_dy + vy * duy_dy + t2 * vn * dun_dy
            dut_dn = vx * dux_dn + vy * duy_dn + t2 * vn * dun_dn

            # scalar expansion rate
            theta = dut_dt + dux_dx + duy_dy + dun_dn + ut / t

            # spatial velocity derivatives
            dvx_dx = (dux_dx - vx * dut_dx) / ut
            dvy_dy = (duy_dy - vy * dut_dy) / ut
            dvn_dn = (dun_dn - vn * dut_dn) / ut
            div_v  = dvx_dx + dvy_dy + dvn_dn

            # fluid acceleration
            at = ut * dut_dt + ux * dut_dx + uy * dut_dy + un * dut_dn + tun * un
            ax = ut * dux_dt + ux * dux_dx + uy * dux_dy + un * dux_dn
            ay = ut * duy_dt + ux * duy_dx + uy * duy_dy + un * duy_dn
            an = ut * dun_dt + ux * dun_dx + uy * dun_dy + un * dun_dn + 2 * ut * un / t

            # velocity-shear tensor
            # constuct spatial projection tensor and traceless-symmetric tensor
            Delta = call_symmetric_traceless_projection(ut, ux, uy, un, t2)

            stt = dut_dt
            stx = (dux_dt - dut_dx) / 2
            sty = (duy_dt - dut_dy) / 2
            stn = (dun_dt - dut_dn / t2) / 2
            sxx = -dux_dx
            sxy = -(dux_dy + duy_dx) / 2
            sxn = -(dux_dn + dun_dx / t2) / 2
            syy = -duy_dy
            syn = -(duy_dn + dun_dy / t2) / 2
            snn = -(dun_dn + ut / t) / t2

            # project out symmetric traceless part
            stt, stx, sty, stn, sxx, sxy, sxn, syy, syn, snn = self.project_sym_tr(Delta, stt, stx, sty, stn, sxx, sxy, sxn, syy, syn, snn)

            # shear-stress and velocity shear contracted term
            pi_sigma = pitt * stt + pixx * sxx + piyy * syy + t4 * pinn * snn + 2 * (pixy * sxy - pitx * stx - pity * sty + t2 * (pixn * sxn + piyn * syn - pitn * stn))
            
            # for shear relaxation equations
            # Christofell symbol contracted terms G^{\mu\nu} = 2 u^\alpha \Gamma^{(\mu}_{\alpha\beta} \pi^{\beta\nu)}
            Gtt = 2 * tun * pitn
            Gtx = tun * pixn
            Gty = tun * piyn
            Gtn = tun * pinn + (ut * pitn + un * pitt) / t
            Gxn = (ut * pixn + un * pitx) / t
            Gyn = (ut * piyn + un * pity) / t
            Gnn = 2 * (ut * pinn + un * pitn) / t
            
            # pi^{\mu\nu} a_\nu terms
            piat = pitt * at - pitx * ax - pity * ay - t2 * pitn * an
            piax = pitx * at - pixx * ax - pixy * ay - t2 * pixn * an
            piay = pity * at - pixy * ax - piyy * ay - t2 * piyn * an
            pian = pitn * at - pixn * ax - piyn * an - t2 * pinn * an
            
            # P^{\mu\nu} = u^{(\mu} \pi{\nu\lambda} a_\lambda
            Ptt = 2 * ut * piat
            Ptx = ut * piax + ux * piat
            Pty = ut * piay + uy * piat
            Ptn = ut * pian + un * piat
            Pxx = 2 * ux * piax
            Pxy = ux * piay + uy * piax
            Pxn = ux * pian + un * piax
            Pyy = 2 * uy * piay
            Pyn = uy * pian + un * piay
            Pnn = 2 * un * pian
            
            # shear relaxation equation
            taupi_inv = 1 / tau_pi
            dpitt = -pitt * taupi_inv - Ptt - Gtt
            dpitx = -pitx * taupi_inv - Ptx - Gtx
            dpity = -pity * taupi_inv - Pty - Gty
            dpitn = -pitn * taupi_inv - Ptn - Gtn
            dpixx = -pixx * taupi_inv - Pxx
            dpixy = -pixx * taupi_inv - Pxy
            dpixn = -pixn * taupi_inv - Pxn - Gxn
            dpiyy = -piyy * taupi_inv - Pyy
            dpiyn = -piyn * taupi_inv - Pyn - Gyn
            dpinn = -pinn * taupi_inv - Pnn - Gnn
            
            # conservation law?
            Tnn = (e + p) * un * un + p / t2 + pinn
            
            # source term
            S = np.zeros_like(q)
            n = 0
            S[n] = -(Ttt / t + t * Tnn) + div_v * (pitt - p) + vx * (dpitt_dx - dp_dx) + vy * (dpitt_dy - dp_dy) + vn * (dpitt_dn - dp_dn) - dpitx_dx  - dpity_dy - dpitn_dn; n += 1
            S[n] = -Ttx / t - dp_dx + div_v * pitx + vx * dpitx_dx + vy * dpitx_dy + vn * dpitx_dn - dpixx_dx  - dpixy_dy - dpixn_dn; n += 1
            S[n] = -Tty / t - dp_dy + div_v * pity + vx * dpity_dx + vy * dpity_dy + vn * dpity_dn - dpixy_dx  - dpiyy_dy - dpiyn_dn; n += 1
            S[n] = -Ttn / t - dp_dn + div_v * pitn + vx * dpitn_dx + vy * dpitn_dy + vn * dpitn_dn - dpixn_dx  - dpiyn_dy - dpinn_dn; n += 1
            
            S[n] = dpitt / ut + div_v * pitt; n += 1
            S[n] = dpitx / ut + div_v * pitx; n += 1
            S[n] = dpity / ut + div_v * pity; n += 1
            S[n] = dpitn / ut + div_v * pitn; n += 1
            S[n] = dpixx / ut + div_v * pixx; n += 1
            S[n] = dpixy / ut + div_v * pixy; n += 1
            S[n] = dpixn / ut + div_v * pixn; n += 1
            S[n] = dpiyy / ut + div_v * piyy; n += 1
            S[n] = dpiyn / ut + div_v * piyn; n += 1
            S[n] = dpinn / ut + div_v * pinn
            return S