# automatically generated by the FlatBuffers compiler, do not modify

# namespace: gamma_api

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class RangeFilter(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RangeFilter()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsRangeFilter(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # RangeFilter
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # RangeFilter
    def Field(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return self._tab.String(o + self._tab.Pos) if o != 0 else None

    # RangeFilter
    def LowerValue(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # RangeFilter
    def LowerValueAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # RangeFilter
    def LowerValueLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return self._tab.VectorLen(o) if o != 0 else 0

    # RangeFilter
    def LowerValueIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # RangeFilter
    def UpperValue(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # RangeFilter
    def UpperValueAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # RangeFilter
    def UpperValueLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return self._tab.VectorLen(o) if o != 0 else 0

    # RangeFilter
    def UpperValueIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # RangeFilter
    def IncludeLower(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # RangeFilter
    def IncludeUpper(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def Start(builder): builder.StartObject(5)
def RangeFilterStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddField(builder, field): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(field), 0)
def RangeFilterAddField(builder, field):
    """This method is deprecated. Please switch to AddField."""
    return AddField(builder, field)
def AddLowerValue(builder, lowerValue): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(lowerValue), 0)
def RangeFilterAddLowerValue(builder, lowerValue):
    """This method is deprecated. Please switch to AddLowerValue."""
    return AddLowerValue(builder, lowerValue)
def StartLowerValueVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def RangeFilterStartLowerValueVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartLowerValueVector(builder, numElems)
def AddUpperValue(builder, upperValue): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(upperValue), 0)
def RangeFilterAddUpperValue(builder, upperValue):
    """This method is deprecated. Please switch to AddUpperValue."""
    return AddUpperValue(builder, upperValue)
def StartUpperValueVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def RangeFilterStartUpperValueVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartUpperValueVector(builder, numElems)
def AddIncludeLower(builder, includeLower): builder.PrependBoolSlot(3, includeLower, 0)
def RangeFilterAddIncludeLower(builder, includeLower):
    """This method is deprecated. Please switch to AddIncludeLower."""
    return AddIncludeLower(builder, includeLower)
def AddIncludeUpper(builder, includeUpper): builder.PrependBoolSlot(4, includeUpper, 0)
def RangeFilterAddIncludeUpper(builder, includeUpper):
    """This method is deprecated. Please switch to AddIncludeUpper."""
    return AddIncludeUpper(builder, includeUpper)
def End(builder): return builder.EndObject()
def RangeFilterEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)