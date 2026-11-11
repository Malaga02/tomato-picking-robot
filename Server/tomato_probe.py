#!/usr/bin/env python3
# Privremena dijagnosticka skripta - NIJE deo trajnog softvera robota.
# Povezuje se na isti websocket server (port 8888) koji koristi i sama
# kontrolna stranica, salje 'tomatoStatus' nekoliko puta zaredom i ispisuje
# sirov odgovor - da vidimo STVARNO stanje (state/lastError/pickedCount)
# TomatoPicker niti, bez oslanjanja na prikaz u browseru.
import asyncio
import websockets
import json
import time

async def main():
    uri = "ws://localhost:8888/"
    async with websockets.connect(uri) as ws:
        await ws.send("admin:123456")
        ack = await ws.recv()
        print("AUTH ACK:", ack)
        for i in range(15):
            await ws.send("tomatoStatus")
            resp = await ws.recv()
            print(f"[{time.strftime('%H:%M:%S')}] {resp}")
            await asyncio.sleep(1)

asyncio.run(main())
