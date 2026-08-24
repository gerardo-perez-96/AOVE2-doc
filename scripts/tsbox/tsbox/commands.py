"""Deshacer/rehacer. Va ANTES que el autoguardado: sin pila de undo,
un borrado accidental está en disco 30 segundos después y no vuelve."""
from __future__ import annotations

from dataclasses import replace

from PySide6.QtGui import QUndoCommand

from .model import GlobalRegion, Mark, Note, Region


class _Base(QUndoCommand):
    def __init__(self, win, text: str):
        super().__init__(text)
        self.win = win

    def touch(self, sid: str | None = None) -> None:
        self.win.session.dirty = True
        self.win.refresh_annotations()
        self.win.refresh_annotation_table()


class AddRegion(_Base):
    def __init__(self, win, region: Region):
        super().__init__(win, f"Añadir región en {region.sid}")
        self.region = region

    def redo(self):
        self.win.session.project.regions.append(self.region)
        self.touch()

    def undo(self):
        self.win.session.project.regions = [
            r for r in self.win.session.project.regions if r.aid != self.region.aid]
        self.touch()


class DeleteRegion(_Base):
    def __init__(self, win, aid: str):
        super().__init__(win, "Borrar región")
        self.region = next(r for r in win.session.project.regions if r.aid == aid)

    def redo(self):
        self.win.session.project.regions = [
            r for r in self.win.session.project.regions if r.aid != self.region.aid]
        self.touch()

    def undo(self):
        self.win.session.project.regions.append(self.region)
        self.touch()


class EditRegion(_Base):
    def __init__(self, win, aid: str, t0: float, t1: float,
                 label: str | None = None):
        super().__init__(win, "Editar región")
        self.aid = aid
        cur = next(r for r in win.session.project.regions if r.aid == aid)
        self.old = replace(cur)
        self.new = replace(cur, t0=t0, t1=t1,
                           label=cur.label if label is None else label)

    def _set(self, val: Region):
        rs = self.win.session.project.regions
        for i, r in enumerate(rs):
            if r.aid == self.aid:
                rs[i] = replace(val)
                break
        self.touch()

    def redo(self):
        self._set(self.new)

    def undo(self):
        self._set(self.old)


class AddGlobalRegion(_Base):
    def __init__(self, win, region: GlobalRegion):
        super().__init__(win, "Añadir zona global")
        self.region = region

    def redo(self):
        self.win.session.project.global_regions.append(self.region)
        self.touch()

    def undo(self):
        self.win.session.project.global_regions = [
            r for r in self.win.session.project.global_regions
            if r.aid != self.region.aid]
        self.touch()


class DeleteGlobalRegion(_Base):
    def __init__(self, win, aid: str):
        super().__init__(win, "Borrar zona global")
        self.region = next(r for r in win.session.project.global_regions
                           if r.aid == aid)

    def redo(self):
        self.win.session.project.global_regions = [
            r for r in self.win.session.project.global_regions
            if r.aid != self.region.aid]
        self.touch()

    def undo(self):
        self.win.session.project.global_regions.append(self.region)
        self.touch()


class EditGlobalRegion(_Base):
    def __init__(self, win, aid: str, t0: float, t1: float,
                 label: str | None = None):
        super().__init__(win, "Editar zona global")
        self.aid = aid
        cur = next(r for r in win.session.project.global_regions if r.aid == aid)
        self.old = replace(cur)
        self.new = replace(cur, t0=t0, t1=t1,
                           label=cur.label if label is None else label)

    def _set(self, val: GlobalRegion):
        rs = self.win.session.project.global_regions
        for i, r in enumerate(rs):
            if r.aid == self.aid:
                rs[i] = replace(val)
                break
        self.touch()

    def redo(self):
        self._set(self.new)

    def undo(self):
        self._set(self.old)


class AddMark(_Base):
    def __init__(self, win, mark: Mark):
        super().__init__(win, "Añadir marca")
        self.mark = mark

    def redo(self):
        self.win.session.project.marks.append(self.mark)
        self.touch()

    def undo(self):
        self.win.session.project.marks = [
            m for m in self.win.session.project.marks if m.aid != self.mark.aid]
        self.touch()


class DeleteMark(_Base):
    def __init__(self, win, aid: str):
        super().__init__(win, "Borrar marca")
        self.mark = next(m for m in win.session.project.marks if m.aid == aid)

    def redo(self):
        self.win.session.project.marks = [
            m for m in self.win.session.project.marks if m.aid != self.mark.aid]
        self.touch()

    def undo(self):
        self.win.session.project.marks.append(self.mark)
        self.touch()


class _NoteBase(_Base):
    def touch(self, sid: str | None = None) -> None:
        # Las notas no tocan paneles ni la tabla de regiones/marcas: solo
        # su propia lista. Reusar touch() de _Base repintaría 18 paneles
        # por cada tecla... no, por cada nota, pero sigue siendo trabajo
        # de sobra que no aporta nada aquí.
        self.win.session.dirty = True
        self.win.refresh_notes()


class AddNote(_NoteBase):
    def __init__(self, win, note: Note):
        super().__init__(win, "Añadir nota")
        self.note = note

    def redo(self):
        self.win.session.project.notes.append(self.note)
        self.touch()

    def undo(self):
        self.win.session.project.notes = [
            n for n in self.win.session.project.notes if n.nid != self.note.nid]
        self.touch()


class DeleteNote(_NoteBase):
    def __init__(self, win, nid: str):
        super().__init__(win, "Borrar nota")
        self.note = next(n for n in win.session.project.notes if n.nid == nid)

    def redo(self):
        self.win.session.project.notes = [
            n for n in self.win.session.project.notes if n.nid != self.note.nid]
        self.touch()

    def undo(self):
        self.win.session.project.notes.append(self.note)
        self.touch()


class EditNote(_NoteBase):
    def __init__(self, win, nid: str, text: str):
        super().__init__(win, "Editar nota")
        self.nid = nid
        cur = next(n for n in win.session.project.notes if n.nid == nid)
        self.old = cur.text
        self.new = text

    def _set(self, text):
        n = next((n for n in self.win.session.project.notes
                  if n.nid == self.nid), None)
        if n is not None:
            n.text = text
        self.touch()

    def redo(self):
        self._set(self.new)

    def undo(self):
        self._set(self.old)


class ReorderSeries(_Base):
    """Mueve una o más series a un nuevo orden global.

    Descubierto por su ausencia: reordenar (por la lista lateral o
    arrastrando el panel) mutaba el proyecto directamente, sin pasar por el
    undo stack -- al contrario que TODO lo demás en este fichero. Un
    arrastre accidental quedaba sin forma de deshacer, justo el tipo de
    error que un click de más produce con más facilidad que borrar algo.
    """

    def __init__(self, win, new_order: list[str]):
        super().__init__(win, "Reordenar series")
        self.new_order = list(new_order)
        self.old_order = [s.sid for s in win.session.project.ordered()]

    def touch(self, sid: str | None = None) -> None:
        self.win.session.dirty = True
        self.win.rebuild_panels()
        self.win.refresh_list()

    def redo(self):
        self.win.session.project.reorder(self.new_order)
        self.touch()

    def undo(self):
        self.win.session.project.reorder(self.old_order)
        self.touch()


class RenameAnnotation(_Base):
    def __init__(self, win, aid: str, label: str):
        super().__init__(win, "Renombrar anotación")
        self.aid = aid
        self.new = label
        obj = win.find_annotation(aid)
        self.old = obj.label

    def _set(self, label):
        obj = self.win.find_annotation(self.aid)
        if obj is not None:
            obj.label = label
        self.touch()

    def redo(self):
        self._set(self.new)

    def undo(self):
        self._set(self.old)
