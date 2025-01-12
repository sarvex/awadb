# automatically generated by the FlatBuffers compiler, do not modify

# namespace: gamma_api

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class FieldInfo(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FieldInfo()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsFieldInfo(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # FieldInfo
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FieldInfo
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return self._tab.String(o + self._tab.Pos) if o != 0 else None

    # FieldInfo
    def DataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # FieldInfo
    def IsIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def Start(builder): builder.StartObject(3)
def FieldInfoStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddName(builder, name): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)
def FieldInfoAddName(builder, name):
    """This method is deprecated. Please switch to AddName."""
    return AddName(builder, name)
def AddDataType(builder, dataType): builder.PrependInt8Slot(1, dataType, 0)
def FieldInfoAddDataType(builder, dataType):
    """This method is deprecated. Please switch to AddDataType."""
    return AddDataType(builder, dataType)
def AddIsIndex(builder, isIndex): builder.PrependBoolSlot(2, isIndex, 0)
def FieldInfoAddIsIndex(builder, isIndex):
    """This method is deprecated. Please switch to AddIsIndex."""
    return AddIsIndex(builder, isIndex)
def End(builder): return builder.EndObject()
def FieldInfoEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)