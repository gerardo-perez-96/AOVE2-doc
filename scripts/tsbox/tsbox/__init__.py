"""tsbox: visor y anotador de series temporales.

Fuerza el binding de Qt que usa pyqtgraph ANTES de que nada lo importe.

pyqtgraph, sin PYQTGRAPH_QT_LIB, prueba los bindings en este orden fijo:
PyQt6, PySide6, PyQt5, PySide2 -- y se queda con el primero que esté
instalado, sin mirar cuál usa el resto de la aplicación. Si en la máquina
hay PyQt6 instalado por CUALQUIER OTRA razón (una extensión de VSCode, otra
herramienta, una dependencia transitiva de otro paquete), pyqtgraph se
engancha a PyQt6 mientras el resto de tsbox usa PySide6. Son dos runtimes
de Qt distintos: sus objetos (QCursor, Qt.PenStyle...) no son intercambiables
entre sí, así que cualquier llamada de pyqtgraph que reciba un tipo de
PySide6 revienta con TypeError.

Reproducido en este entorno instalando PyQt6 junto a PySide6: sin esta línea,
la aplicación no llega ni a abrir la ventana -- revienta en panel.py al
definir el pen de los huecos, antes de pintar nada. Con la variable puesta,
pyqtgraph se ata a PySide6 pase lo que pase.

Tiene que ejecutarse aquí, en el __init__.py del paquete: es el único punto
por el que pasan TODOS los caminos de entrada (app.py, los tests que
importan tsbox.mainwindow o tsbox.session directamente, etc.) antes de que
cualquier submódulo tenga ocasión de hacer `import pyqtgraph`.
"""
import os as _os

_os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
