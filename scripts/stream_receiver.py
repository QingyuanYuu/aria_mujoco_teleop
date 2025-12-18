"""
Stream receiver module for Aria device.

This module handles the HTTP server that receives hand tracking streams
from Meta Aria glasses.
"""
import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver
from hand_tracking import hand_cb


def start_receiver(host: str, port: int):
    """
    Start the HTTP server to receive hand tracking streams from Aria device.
    
    Args:
        host: Server host address (e.g., "0.0.0.0")
        port: Server port number (e.g., 6768)
    
    Returns:
        StreamReceiver instance
    """
    cfg = sdk_gen2.HttpServerConfig()
    cfg.address = host
    cfg.port = port

    sr = receiver.StreamReceiver(enable_image_decoding=False, enable_raw_stream=False)
    sr.set_server_config(cfg)
    sr.register_hand_pose_callback(hand_cb)

    print(f"[Receiver] Listening on {host}:{port} ... (start your device streaming)")
    sr.start_server()
    return sr

