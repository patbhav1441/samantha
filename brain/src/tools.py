import os
import asyncio
from typing import Dict, Any, Optional

from meross_iot.http_api import MerossHttpClient
from meross_iot.manager import MerossManager


async def _garage_action(action: str, door: Optional[str]) -> Dict[str, Any]:
    """
    Connect to Meross cloud, discover MSG100 devices, and open/close selected door.

    door: "left", "right", or None (None = first found)
    """
    email = os.environ.get("MEROSS_EMAIL")
    password = os.environ.get("MEROSS_PASSWORD")
    if not email or not password:
        return {"ok": False, "error": "MEROSS_EMAIL and MEROSS_PASSWORD must be set."}

    api_base_url = os.environ.get("MEROSS_API_BASE_URL", "https://iot.meross.com")

    http_client = await MerossHttpClient.async_from_user_password(
        email=email,
        password=password,
        api_base_url=api_base_url,
    )
    mgr = MerossManager(http_client=http_client)

    try:
        await mgr.async_init()
        await mgr.async_device_discovery()

        garages = mgr.find_devices(device_type="msg100")

        if not garages:
            all_devices = mgr.find_devices()
            garages = [
                d for d in all_devices
                if (d.device_type or "").lower().startswith("msg100")
                or "garage" in (d.device_type or "").lower()
            ]

        if not garages:
            return {"ok": False, "error": "No Meross MSG100 garage devices found."}

        selected = None
        door_norm = door.lower() if door else None

        if door_norm in ("left", "right"):
            for g in garages:
                name = (getattr(g, "name", "") or "").lower()
                if door_norm in name:
                    selected = g
                    break

        if selected is None:
            selected = garages[0]

        garage = selected
        await garage.async_update()

        if action == "open":
            await garage.async_open()
        elif action == "close":
            await garage.async_close()
        else:
            return {"ok": False, "error": f"Unknown action: {action}"}

        await garage.async_update()

        state_val = None
        get_state = getattr(garage, "get_current_state", None)
        if callable(get_state):
            try:
                state_val = get_state()
            except Exception:
                state_val = None

        return {
            "ok": True,
            "device_name": getattr(garage, "name", "garage"),
            "requested_door": door_norm,
            "action": action,
            "state": state_val,
        }

    finally:
        mgr.close()
        await http_client.async_logout()


def tool_garage_open(door: Optional[str] = None) -> Dict[str, Any]:
    return asyncio.run(_garage_action("open", door))


def tool_garage_close(door: Optional[str] = None) -> Dict[str, Any]:
    return asyncio.run(_garage_action("close", door))

