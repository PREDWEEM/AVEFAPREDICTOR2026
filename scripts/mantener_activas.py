"""Visita las apps públicas de Streamlit del repositorio para que no entren en hibernación.

Streamlit Community Cloud suspende una app después de 12 horas sin visitas. GitHub Actions ejecuta este script cada
cuatro horas (las ejecuciones programadas pueden demorarse varias horas). Cada visita abre la app en Chromium, pulsa
el botón para despertarla si está suspendida y espera hasta que la interfaz esté cargada de verdad (título y
controles), no solo una respuesta HTTP. Si alguna app no carga en cinco minutos o muestra una excepción, la ejecución
falla y GitHub lo avisa según la configuración de notificaciones. No modifica datos.

Las direcciones pueden reemplazarse con la variable de entorno APPS_STREAMLIT (JSON: lista de objetos con
``nombre``, ``url`` y ``titulo``).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from playwright.sync_api import Error, TimeoutError, sync_playwright

APPS = [
    {"nombre": "AVEFA Pulsos", "url": "https://apppulsospy-uad2v3sugxzwu9n7dkwwqi.streamlit.app/",
     "titulo": "AVEFA Pulsos"},
    {"nombre": "PREDWEEM v8.5", "url": "https://apppredweemv8py-gfl8smqwmnkwhccwsaqvhd.streamlit.app/",
     "titulo": "PREDWEEM v8.5"},
]
ESPERA_MAXIMA_S = 300
BOTONES_DESPERTAR = ("Yes, get this app back up!", "Wake up")


def _lista_apps() -> list[dict]:
    texto = os.environ.get("APPS_STREAMLIT", "").strip()
    return json.loads(texto) if texto else APPS


def _app_lista(frame, titulo: str) -> bool:
    """La app está cargada si su título (h1) y la barra lateral están visibles en el mismo documento."""
    encabezado = frame.locator("h1").filter(has_text=titulo).first
    return encabezado.is_visible() and frame.locator('[data-testid="stSidebar"]').first.is_visible()


def visitar(app: dict, navegador) -> str:
    pagina = navegador.new_page()
    pagina.set_default_timeout(5000)
    inicio = time.monotonic()
    desperto = False
    lista_desde = None
    try:
        try:
            pagina.goto(app["url"], wait_until="domcontentloaded", timeout=60000)
        except TimeoutError:
            print(f"{app['nombre']}: la apertura inicial sigue en curso.", flush=True)
        while time.monotonic() - inicio < ESPERA_MAXIMA_S:
            lista = False
            # Community Cloud aloja la interfaz dentro de un iframe
            for frame in pagina.frames:
                try:
                    if frame.locator('[data-testid="stException"]').count():
                        raise RuntimeError(f"{app['nombre']}: la app muestra una excepción de Streamlit.")
                    for texto in BOTONES_DESPERTAR:
                        boton = frame.get_by_role("button", name=texto)
                        if not desperto and boton.count() and boton.first.is_visible() and boton.first.is_enabled():
                            boton.first.click()
                            desperto = True
                            print(f"{app['nombre']}: estaba suspendida; se pidió despertarla.", flush=True)
                    lista = _app_lista(frame, app["titulo"]) or lista
                except Error:
                    continue  # un iframe puede cambiar mientras Streamlit arranca
            if lista:
                lista_desde = lista_desde or time.monotonic()
                if time.monotonic() - lista_desde >= 5:
                    seg = round(time.monotonic() - inicio)
                    return f"{app['nombre']}: operativa ({'despertada, ' if desperto else ''}{seg} s)."
            else:
                lista_desde = None
            pagina.wait_for_timeout(5000)
        raise RuntimeError(f"{app['nombre']}: no se pudo confirmar la carga en {ESPERA_MAXIMA_S // 60} minutos.")
    finally:
        pagina.close()


def informar(lineas: list[str]) -> None:
    texto = "\n".join(lineas)
    print(texto, flush=True)
    if resumen := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(resumen).open("a", encoding="utf-8") as f:
            f.write(texto.replace("\n", "\n\n") + "\n")


def main() -> int:
    lineas, fallo = [], False
    with sync_playwright() as pw:
        navegador = pw.chromium.launch()
        try:
            for app in _lista_apps():
                try:
                    lineas.append(visitar(app, navegador))
                except (RuntimeError, Error) as error:
                    fallo = True
                    lineas.append(f"FALLO {error}" if isinstance(error, RuntimeError) else f"FALLO {app['nombre']}: {error}")
        finally:
            navegador.close()
    informar(lineas)
    return 1 if fallo else 0


if __name__ == "__main__":
    raise SystemExit(main())
