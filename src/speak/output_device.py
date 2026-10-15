"""Watch the macOS default output without relying on PortAudio's cached default."""

import ctypes as C
from contextlib import contextmanager
import sys
from threading import Event

import sounddevice as sd


class PropertyAddress(C.Structure):
    _fields_ = [("selector", C.c_uint32), ("scope", C.c_uint32), ("element", C.c_uint32)]


def check_status(status, operation):
    if status:
        raise RuntimeError(f"Core Audio {operation} failed: {status}")


@contextmanager
def output_device_changes(enabled=True):
    changed = Event()
    if not enabled or sys.platform != "darwin":
        yield changed
        return
    library = C.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    address = PropertyAddress(int.from_bytes(b"dOut", "big"), int.from_bytes(b"glob", "big"), 0)
    callback_type = C.CFUNCTYPE(C.c_int32, C.c_uint32, C.c_uint32,
                              C.POINTER(PropertyAddress), C.c_void_p)
    signature = [C.c_uint32, C.POINTER(PropertyAddress), callback_type, C.c_void_p]
    for name in ("AudioObjectAddPropertyListener", "AudioObjectRemovePropertyListener"):
        function = getattr(library, name)
        function.argtypes = signature
        function.restype = C.c_int32

    @callback_type
    def callback(device, count, addresses, context):
        changed.set()
        return 0

    check_status(library.AudioObjectAddPropertyListener(1, C.byref(address), callback, None),
                 "register output listener")
    try:
        yield changed
    finally:
        check_status(library.AudioObjectRemovePropertyListener(1, C.byref(address), callback, None),
                     "remove output listener")


def refresh_output_device():
    # PortAudio snapshots devices at initialization. No stream can remain open here.
    sd._terminate()
    sd._initialize()
    return sd.query_devices(kind="output")["name"]
