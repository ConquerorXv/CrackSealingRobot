from __future__ import annotations

import importlib
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


class SimulatedArm:
    """Six-coordinate test model, not a physical Unitree dynamics simulator."""
    def __init__(self):
        self.q = np.zeros(6)
        self.history = []

    def fk(self, q):
        transform = np.eye(4)
        transform[:3, 3] = q[:3]
        transform[:3, :3] = Rotation.from_euler('xyz', q[3:]).as_matrix()
        return transform

    def jacobian(self, q):
        jac = np.zeros((6, 6))
        jac[:3, :3] = np.eye(3)
        rotation = self.fk(q)[:3, :3]
        for j in range(3, 6):
            shifted = np.array(q, float)
            shifted[j] += 1e-6
            jac[3:, j] = Rotation.from_matrix(self.fk(shifted)[:3, :3]@rotation.T).as_rotvec()/1e-6
        return jac

    def joint_limits(self):
        return np.full(6, -10.), np.full(6, 10.)

    def speed_limits(self):
        return np.full(6, 10.)

    def current_q(self):
        return self.q.copy()

    def flange_pose(self):
        return self.fk(self.q)

    def send(self, q, qd):
        self.q = np.asarray(q).copy()
        self.history.append(self.q.copy())

    def close(self):
        pass


class SimulatedPump:
    def __init__(self):
        self.history = []

    def set_speed(self, rpm):
        self.history.append(('speed', float(rpm)))

    def forward(self):
        self.history.append(('forward',))

    def reverse(self):
        self.history.append(('reverse',))

    def stop(self):
        self.history.append(('stop',))

    def close(self):
        pass


class UnitreeArm:
    def __init__(self, settings):
        sdk = Path(settings['sdk_path'])
        if not sdk.is_dir():
            raise FileNotFoundError(f'Unitree SDK directory not found: {sdk}')
        sys.path.insert(0, str(sdk))
        module = importlib.import_module('unitree_arm_interface')
        self.arm = module.ArmInterface(hasGripper=False)
        self.model = self.arm._ctrlComp.armModel
        self.communication_dt = float(settings['sdk_dt_s'])
        self.arm.loopOn()
        time.sleep(.1)
        self.arm.startTrack(module.ArmFSMState.JOINTCTRL)
        self.arm.loopOff()
        self.arm.sendRecv()
        sdk_dt = float(self.arm._ctrlComp.dt)
        if abs(sdk_dt-self.communication_dt) > 1e-5:
            raise ValueError(f'Configured SDK period {self.communication_dt} differs from SDK {sdk_dt}')
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error = None
        self._target_q = np.asarray(self.arm.lowstate.getQ()).copy()
        self._target_qd = np.zeros(6)
        self._thread = threading.Thread(target=self._communication_loop, daemon=True)
        self._thread.start()

    def _communication_loop(self):
        deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                with self._lock:
                    self.arm.setArmCmd(self._target_q, self._target_qd, np.zeros(6))
                    self.arm.sendRecv()
                deadline += self.communication_dt
                if time.monotonic()-deadline > .1:
                    raise TimeoutError('Unitree SDK communication deadline exceeded')
                self._stop.wait(max(0., deadline-time.monotonic()))
            except BaseException as error:
                self._error = error
                self._stop.set()

    def _check_connection(self):
        if self._error is not None:
            raise ConnectionError(f'Unitree communication failed: {self._error}') from self._error

    def fk(self, q):
        return np.asarray(self.model.forwardKinematics(q, 6))

    def jacobian(self, q):
        return np.asarray(self.model.CalcJacobian(q))

    def joint_limits(self):
        return np.asarray(self.model.getJointQMin()), np.asarray(self.model.getJointQMax())

    def speed_limits(self):
        return np.asarray(self.model.getJointSpeedMax())

    def current_q(self):
        self._check_connection()
        with self._lock:
            return np.asarray(self.arm.lowstate.getQ()).copy()

    def flange_pose(self):
        return self.fk(self.current_q())

    def send(self, q, qd):
        self._check_connection()
        with self._lock:
            self._target_q = np.asarray(q).copy()
            self._target_qd = np.asarray(qd).copy()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.)
        self.arm.loopOff()


class ModbusPump:
    def __init__(self, settings):
        self.settings = settings
        client_module = importlib.import_module('pymodbus.client')
        self.client = client_module.ModbusSerialClient(
            port=settings['port'], baudrate=settings.get('baudrate', 9600),
            bytesize=8, parity='N', stopbits=1, timeout=settings.get('timeout_s', 1.))
        if not self.client.connect():
            raise ConnectionError('Cannot connect to pump Modbus port')
        try:
            self.stop()
            for address, value in settings.get('initialization_registers', []):
                self.write(address, value)
        except BaseException:
            self.client.close()
            raise

    def write(self, address, value):
        # The hardware extra pins pymodbus 3.6, whose device keyword is slave.
        reply = self.client.write_register(int(address), int(value), slave=int(self.settings['slave']))
        if reply is None or reply.isError():
            raise IOError(f'Pump Modbus write failed at register {address}')

    def set_speed(self, rpm):
        value = int(round(rpm*float(self.settings['register_units_per_rpm'])))
        if not 0 <= value <= 65535:
            raise ValueError('Pump speed exceeds a 16-bit register')
        self.write(self.settings['speed_register'], value)

    def forward(self):
        self.write(self.settings['operation_register'], self.settings['forward_value'])

    def reverse(self):
        self.write(self.settings['operation_register'], self.settings['reverse_value'])

    def stop(self):
        self.write(self.settings['operation_register'], self.settings['stop_value'])

    def close(self):
        self.client.close()
