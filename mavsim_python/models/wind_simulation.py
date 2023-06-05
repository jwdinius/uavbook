"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
import sys
sys.path.append('..')
from tools.transfer_function import TransferFunction
import numpy as np
from enum import Enum

LOW_ALT = int(50)
MED_ALT = int(600)


def closest_altitude(altitude):
    if abs(altitude - LOW_ALT) <= abs(altitude - MED_ALT):
        return LOW_ALT
    else:
        return MED_ALT

class Turbulence(Enum):
    LOW=0,
    MED=1


class WindSimulation:
    def __init__(self, Ts, gust_flag = True, steady_state = np.array([[0., 0., 0.]]).T, turbulence=Turbulence.LOW):
        # steady state wind defined in the inertial frame
        self._steady_state = steady_state
        ##### TODO #####
        # XXX Assume that turbulence level is set at initialization
        self._turbulence = turbulence
        #   Dryden gust model parameters (pg 56 UAV book)
        self._dryden_params = {
            LOW_ALT: {
                "L_u": 200.,
                "L_v": 200.,
                "L_w": 50.,
                "sigma_u": 1.06,
                "sigma_v": 1.06,
                "sigma_w": 0.7
            },
            MED_ALT: {
                "L_u": 533.,
                "L_v": 533.,
                "L_w": 533.,
                "sigma_u": 1.5,
                "sigma_v": 1.5,
                "sigma_w": 1.5
            },
        }

        # initialize at low altitude
        self.update_params(float(LOW_ALT))

        # Dryden transfer functions (section 4.4 UAV book) - Fill in proper num and den
        self._closest_altitude = LOW_ALT
        #self.u_w = TransferFunction(num=np.array([[0]]), den=np.array([[1,1]]),Ts=Ts)
        #self.v_w = TransferFunction(num=np.array([[0,0]]), den=np.array([[1,1,1]]),Ts=Ts)
        #self.w_w = TransferFunction(num=np.array([[0,0]]), den=np.array([[1,1,1]]),Ts=Ts)
        self._Ts = Ts

    def update_params(self, altitude):
        if self._turbulence == Turbulence.LOW:
            multiplier = 1.
        else:
            multiplier = 2.
        
        closest_alt = closest_altitude(altitude)
        
        params = self._dryden_params[closest_alt]
        self._Lu = params["L_u"] 
        self._Lv = params["L_v"]
        self._Lw = params["L_w"]
        self._sigu = params["sigma_u"] * multiplier
        self._sigv = params["sigma_v"] * multiplier
        self._sigw = params["sigma_w"] * multiplier


    def update(self, altitude, airspeed):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        closest_alt = closest_altitude(altitude)
        if self._closest_altitude != closest_alt:
            # XXX transition happened, so update the gust model
            self.update_params(altitude)
            self._closest_altitude = closest_alt
        airspeed_div_Lu = airspeed / self._Lu
        u_scalar = self._sigu * np.sqrt(2. * airspeed_div_Lu)
        self.u_w = TransferFunction(num=np.array([[u_scalar]]), den=np.array([[1, airspeed_div_Lu]]), Ts=self._Ts)
        airspeed_div_Lv = airspeed / self._Lv
        v_scalar = self._sigv * np.sqrt(3. * airspeed_div_Lv)
        self.v_w = TransferFunction(num=np.array([[v_scalar, v_scalar * airspeed_div_Lv / np.sqrt(3)]]), den=np.array([[1, 2. * airspeed_div_Lv, airspeed_div_Lv**2]]), Ts=self._Ts)
        airspeed_div_Lw = airspeed / self._Lv
        w_scalar = self._sigw * np.sqrt(3. * airspeed_div_Lw)
        self.w_w = TransferFunction(num=np.array([[w_scalar, w_scalar * airspeed_div_Lw / np.sqrt(3)]]), den=np.array([[1, 2. * airspeed_div_Lw, airspeed_div_Lw**2]]), Ts=self._Ts)
        
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])

        return np.concatenate(( self._steady_state, gust ))

