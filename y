ExtractBasicBlocks for System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
Split node 1 at instruction IL_000b: ldarg.0
Node prior to split:

Node: 1
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Instructions:
        IL_0000: nop
        IL_0001: ldarg.1
        IL_0002: ldc.i4.2
        IL_0003: rem
        IL_0004: ldc.i4.0
        IL_0005: ceq
        IL_0007: stloc.0
        IL_0008: ldloc.0
        IL_0009: brfalse.s IL_001e
        IL_000b: ldarg.0
        IL_000c: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0011: ldarg.1
        IL_0012: ldarg.1
        IL_0013: ldc.i4.s 20
        IL_0015: mul
        IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_001b: nop
        IL_001c: br.s IL_002f
        IL_001e: ldarg.0
        IL_001f: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0024: ldarg.1
        IL_0025: ldarg.1
        IL_0026: ldc.i4.s 30
        IL_0028: mul
        IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_002e: nop
        IL_002f: ret

New node is 2
Node after split:

Node: 1
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Instructions:
        IL_0000: nop
        IL_0001: ldarg.1
        IL_0002: ldc.i4.2
        IL_0003: rem
        IL_0004: ldc.i4.0
        IL_0005: ceq
        IL_0007: stloc.0
        IL_0008: ldloc.0
        IL_0009: brfalse.s IL_001e

Newly created node:

Node: 2
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Instructions:
        IL_000b: ldarg.0
        IL_000c: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0011: ldarg.1
        IL_0012: ldarg.1
        IL_0013: ldc.i4.s 20
        IL_0015: mul
        IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_001b: nop
        IL_001c: br.s IL_002f
        IL_001e: ldarg.0
        IL_001f: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0024: ldarg.1
        IL_0025: ldarg.1
        IL_0026: ldc.i4.s 30
        IL_0028: mul
        IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_002e: nop
        IL_002f: ret


Split node 2 at instruction IL_001e: ldarg.0
Node prior to split:

Node: 2
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Instructions:
        IL_000b: ldarg.0
        IL_000c: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0011: ldarg.1
        IL_0012: ldarg.1
        IL_0013: ldc.i4.s 20
        IL_0015: mul
        IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_001b: nop
        IL_001c: br.s IL_002f
        IL_001e: ldarg.0
        IL_001f: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0024: ldarg.1
        IL_0025: ldarg.1
        IL_0026: ldc.i4.s 30
        IL_0028: mul
        IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_002e: nop
        IL_002f: ret

New node is 3
Node after split:

Node: 2
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Instructions:
        IL_000b: ldarg.0
        IL_000c: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0011: ldarg.1
        IL_0012: ldarg.1
        IL_0013: ldc.i4.s 20
        IL_0015: mul
        IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_001b: nop
        IL_001c: br.s IL_002f

Newly created node:

Node: 3
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Instructions:
        IL_001e: ldarg.0
        IL_001f: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0024: ldarg.1
        IL_0025: ldarg.1
        IL_0026: ldc.i4.s 30
        IL_0028: mul
        IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_002e: nop
        IL_002f: ret


Split node 3 at instruction IL_002f: ret
Node prior to split:

Node: 3
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Instructions:
        IL_001e: ldarg.0
        IL_001f: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0024: ldarg.1
        IL_0025: ldarg.1
        IL_0026: ldc.i4.s 30
        IL_0028: mul
        IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_002e: nop
        IL_002f: ret

New node is 4
Node after split:

Node: 3
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Instructions:
        IL_001e: ldarg.0
        IL_001f: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0024: ldarg.1
        IL_0025: ldarg.1
        IL_0026: ldc.i4.s 30
        IL_0028: mul
        IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_002e: nop

Newly created node:

Node: 4
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Instructions:
        IL_002f: ret


digraph {
1 -> 3;
1 -> 2;
2 -> 4;
3 -> 4;
}

Graph:

List of entries blocks:
    Node    Method
       1    System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)

List of callers:
    Node    Instruction
       1    IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       1    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       2    IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       2    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       3    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)

List of orphan blocks:
    Node    Method


Node: 1
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges to: 3 2
    Instructions:
        IL_0000: nop
        IL_0001: ldarg.1
        IL_0002: ldc.i4.2
        IL_0003: rem
        IL_0004: ldc.i4.0
        IL_0005: ceq
        IL_0007: stloc.0
        IL_0008: ldloc.0
        IL_0009: brfalse.s IL_001e


Node: 2
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 1
    Edges to: 4
    Instructions:
        IL_000b: ldarg.0
        IL_000c: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0011: ldarg.1
        IL_0012: ldarg.1
        IL_0013: ldc.i4.s 20
        IL_0015: mul
        IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_001b: nop
        IL_001c: br.s IL_002f


Node: 3
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 1
    Edges to: 4
    Instructions:
        IL_001e: ldarg.0
        IL_001f: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0024: ldarg.1
        IL_0025: ldarg.1
        IL_0026: ldc.i4.s 30
        IL_0028: mul
        IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_002e: nop


Node: 4
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 2 3
    Instructions:
        IL_002f: ret


ExtractBasicBlocks for System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
Split node 5 at instruction IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
Node prior to split:

Node: 5
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Instructions:
        IL_0000: ldarg.1
        IL_0001: ldarg.0
        IL_0002: ldfld System.Int32 System.Collections.Generic.List`1<T>::_size
        IL_0007: blt.un.s IL_000e
        IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
        IL_000e: ldarg.0
        IL_000f: ldfld T[] System.Collections.Generic.List`1<T>::_items
        IL_0014: ldarg.1
        IL_0015: ldarg.2
        IL_0016: stelem.any T
        IL_001b: ldarg.0
        IL_001c: ldarg.0
        IL_001d: ldfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0022: ldc.i4.1
        IL_0023: add
        IL_0024: stfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0029: ret

New node is 6
Node after split:

Node: 5
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Instructions:
        IL_0000: ldarg.1
        IL_0001: ldarg.0
        IL_0002: ldfld System.Int32 System.Collections.Generic.List`1<T>::_size
        IL_0007: blt.un.s IL_000e

Newly created node:

Node: 6
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Instructions:
        IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
        IL_000e: ldarg.0
        IL_000f: ldfld T[] System.Collections.Generic.List`1<T>::_items
        IL_0014: ldarg.1
        IL_0015: ldarg.2
        IL_0016: stelem.any T
        IL_001b: ldarg.0
        IL_001c: ldarg.0
        IL_001d: ldfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0022: ldc.i4.1
        IL_0023: add
        IL_0024: stfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0029: ret


Split node 6 at instruction IL_000e: ldarg.0
Node prior to split:

Node: 6
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Instructions:
        IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
        IL_000e: ldarg.0
        IL_000f: ldfld T[] System.Collections.Generic.List`1<T>::_items
        IL_0014: ldarg.1
        IL_0015: ldarg.2
        IL_0016: stelem.any T
        IL_001b: ldarg.0
        IL_001c: ldarg.0
        IL_001d: ldfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0022: ldc.i4.1
        IL_0023: add
        IL_0024: stfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0029: ret

New node is 7
Node after split:

Node: 6
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Instructions:
        IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()

Newly created node:

Node: 7
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Instructions:
        IL_000e: ldarg.0
        IL_000f: ldfld T[] System.Collections.Generic.List`1<T>::_items
        IL_0014: ldarg.1
        IL_0015: ldarg.2
        IL_0016: stelem.any T
        IL_001b: ldarg.0
        IL_001c: ldarg.0
        IL_001d: ldfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0022: ldc.i4.1
        IL_0023: add
        IL_0024: stfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0029: ret


digraph {
1 -> 3;
1 -> 2;
2 -> 4;
3 -> 4;
5 -> 7;
5 -> 6;
6 -> 7;
}

Graph:

List of entries blocks:
    Node    Method
       1    System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
       5    System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)

List of callers:
    Node    Instruction
       1    IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       1    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       2    IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       2    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       3    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       5    IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
       6    IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()

List of orphan blocks:
    Node    Method


Node: 1
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges to: 3 2
    Instructions:
        IL_0000: nop
        IL_0001: ldarg.1
        IL_0002: ldc.i4.2
        IL_0003: rem
        IL_0004: ldc.i4.0
        IL_0005: ceq
        IL_0007: stloc.0
        IL_0008: ldloc.0
        IL_0009: brfalse.s IL_001e


Node: 2
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 1
    Edges to: 4
    Instructions:
        IL_000b: ldarg.0
        IL_000c: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0011: ldarg.1
        IL_0012: ldarg.1
        IL_0013: ldc.i4.s 20
        IL_0015: mul
        IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_001b: nop
        IL_001c: br.s IL_002f


Node: 3
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 1
    Edges to: 4
    Instructions:
        IL_001e: ldarg.0
        IL_001f: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0024: ldarg.1
        IL_0025: ldarg.1
        IL_0026: ldc.i4.s 30
        IL_0028: mul
        IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_002e: nop


Node: 4
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 2 3
    Instructions:
        IL_002f: ret


Node: 5
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges to: 7 6
    Instructions:
        IL_0000: ldarg.1
        IL_0001: ldarg.0
        IL_0002: ldfld System.Int32 System.Collections.Generic.List`1<T>::_size
        IL_0007: blt.un.s IL_000e


Node: 6
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges from: 5
    Edges to: 7
    Instructions:
        IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()


Node: 7
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges from: 5 6
    Instructions:
        IL_000e: ldarg.0
        IL_000f: ldfld T[] System.Collections.Generic.List`1<T>::_items
        IL_0014: ldarg.1
        IL_0015: ldarg.2
        IL_0016: stelem.any T
        IL_001b: ldarg.0
        IL_001c: ldarg.0
        IL_001d: ldfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0022: ldc.i4.1
        IL_0023: add
        IL_0024: stfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0029: ret


ExtractBasicBlocks for System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
digraph {
1 -> 3;
1 -> 2;
2 -> 4;
3 -> 4;
5 -> 7;
5 -> 6;
6 -> 7;
8;
}

Graph:

List of entries blocks:
    Node    Method
       1    System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
       5    System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       8    System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()

List of callers:
    Node    Instruction
       1    IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       1    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       2    IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       2    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       3    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       5    IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
       6    IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()

List of orphan blocks:
    Node    Method


Node: 1
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges to: 3 2
    Instructions:
        IL_0000: nop
        IL_0001: ldarg.1
        IL_0002: ldc.i4.2
        IL_0003: rem
        IL_0004: ldc.i4.0
        IL_0005: ceq
        IL_0007: stloc.0
        IL_0008: ldloc.0
        IL_0009: brfalse.s IL_001e


Node: 2
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 1
    Edges to: 4
    Instructions:
        IL_000b: ldarg.0
        IL_000c: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0011: ldarg.1
        IL_0012: ldarg.1
        IL_0013: ldc.i4.s 20
        IL_0015: mul
        IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_001b: nop
        IL_001c: br.s IL_002f


Node: 3
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 1
    Edges to: 4
    Instructions:
        IL_001e: ldarg.0
        IL_001f: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0024: ldarg.1
        IL_0025: ldarg.1
        IL_0026: ldc.i4.s 30
        IL_0028: mul
        IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_002e: nop


Node: 4
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 2 3
    Instructions:
        IL_002f: ret


Node: 5
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges to: 7 6
    Instructions:
        IL_0000: ldarg.1
        IL_0001: ldarg.0
        IL_0002: ldfld System.Int32 System.Collections.Generic.List`1<T>::_size
        IL_0007: blt.un.s IL_000e


Node: 6
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges from: 5
    Edges to: 7
    Instructions:
        IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()


Node: 7
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges from: 5 6
    Instructions:
        IL_000e: ldarg.0
        IL_000f: ldfld T[] System.Collections.Generic.List`1<T>::_items
        IL_0014: ldarg.1
        IL_0015: ldarg.2
        IL_0016: stelem.any T
        IL_001b: ldarg.0
        IL_001c: ldarg.0
        IL_001d: ldfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0022: ldc.i4.1
        IL_0023: add
        IL_0024: stfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0029: ret


Node: 8
    Method System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
    HasThis   False
    Args   0
    Locals 0
    Return (reuse) False
    Instructions:
        IL_0000: nop
        IL_0001: ret


Graph:

List of entries blocks:
    Node    Method
       1    System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
       5    System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       8    System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()

List of callers:
    Node    Instruction
       1    IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       1    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       2    IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       2    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       3    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       5    IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
       6    IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()

List of orphan blocks:
    Node    Method


Node: 1
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges to: 3 2
    Instructions:
        IL_0000: nop
        IL_0001: ldarg.1
        IL_0002: ldc.i4.2
        IL_0003: rem
        IL_0004: ldc.i4.0
        IL_0005: ceq
        IL_0007: stloc.0
        IL_0008: ldloc.0
        IL_0009: brfalse.s IL_001e


Node: 2
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 1
    Edges to: 4
    Instructions:
        IL_000b: ldarg.0
        IL_000c: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0011: ldarg.1
        IL_0012: ldarg.1
        IL_0013: ldc.i4.s 20
        IL_0015: mul
        IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_001b: nop
        IL_001c: br.s IL_002f


Node: 3
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 1
    Edges to: 4
    Instructions:
        IL_001e: ldarg.0
        IL_001f: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0024: ldarg.1
        IL_0025: ldarg.1
        IL_0026: ldc.i4.s 30
        IL_0028: mul
        IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_002e: nop


Node: 4
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 2 3
    Instructions:
        IL_002f: ret


Node: 5
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges to: 7 6
    Instructions:
        IL_0000: ldarg.1
        IL_0001: ldarg.0
        IL_0002: ldfld System.Int32 System.Collections.Generic.List`1<T>::_size
        IL_0007: blt.un.s IL_000e


Node: 6
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges from: 5
    Edges to: 7
    Instructions:
        IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()


Node: 7
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges from: 5 6
    Instructions:
        IL_000e: ldarg.0
        IL_000f: ldfld T[] System.Collections.Generic.List`1<T>::_items
        IL_0014: ldarg.1
        IL_0015: ldarg.2
        IL_0016: stelem.any T
        IL_001b: ldarg.0
        IL_001c: ldarg.0
        IL_001d: ldfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0022: ldc.i4.1
        IL_0023: add
        IL_0024: stfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0029: ret


Node: 8
    Method System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
    HasThis   False
    Args   0
    Locals 0
    Return (reuse) False
    Instructions:
        IL_0000: nop
        IL_0001: ret

digraph {
1 -> 3;
1 -> 2;
2 -> 4;
3 -> 4;
5 -> 7;
5 -> 6;
6 -> 7;
8;
}

discovery     00:00:00.9161993
Considering 1
Considering 2
Considering 3
Considering 4
Considering 5
Considering 6
Considering 7
Considering 8
Graph:

List of entries blocks:
    Node    Method
       1    System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
       5    System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       8    System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()

List of callers:
    Node    Instruction
       1    IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       1    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       2    IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       2    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       3    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       5    IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
       6    IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()

List of orphan blocks:
    Node    Method


Node: 1
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges to: 3 2
    Instructions:
        IL_0000: nop
        IL_0001: ldarg.1
        IL_0002: ldc.i4.2
        IL_0003: rem
        IL_0004: ldc.i4.0
        IL_0005: ceq
        IL_0007: stloc.0
        IL_0008: ldloc.0
        IL_0009: brfalse.s IL_001e


Node: 2
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 1
    Edges to: 4
    Instructions:
        IL_000b: ldarg.0
        IL_000c: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0011: ldarg.1
        IL_0012: ldarg.1
        IL_0013: ldc.i4.s 20
        IL_0015: mul
        IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_001b: nop
        IL_001c: br.s IL_002f


Node: 3
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 1
    Edges to: 4
    Instructions:
        IL_001e: ldarg.0
        IL_001f: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0024: ldarg.1
        IL_0025: ldarg.1
        IL_0026: ldc.i4.s 30
        IL_0028: mul
        IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_002e: nop


Node: 4
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 2 3
    Instructions:
        IL_002f: ret


Node: 5
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges to: 7 6
    Instructions:
        IL_0000: ldarg.1
        IL_0001: ldarg.0
        IL_0002: ldfld System.Int32 System.Collections.Generic.List`1<T>::_size
        IL_0007: blt.un.s IL_000e


Node: 6
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges from: 5
    Edges to: 7
    Instructions:
        IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()


Node: 7
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges from: 5 6
    Instructions:
        IL_000e: ldarg.0
        IL_000f: ldfld T[] System.Collections.Generic.List`1<T>::_items
        IL_0014: ldarg.1
        IL_0015: ldarg.2
        IL_0016: stelem.any T
        IL_001b: ldarg.0
        IL_001c: ldarg.0
        IL_001d: ldfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0022: ldc.i4.1
        IL_0023: add
        IL_0024: stfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0029: ret


Node: 8
    Method System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
    HasThis   False
    Args   0
    Locals 0
    Return (reuse) False
    Instructions:
        IL_0000: nop
        IL_0001: ret

Graph:

List of entries blocks:
    Node    Method
       1    System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
       5    System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       8    System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()

List of callers:
    Node    Instruction
       1    IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       1    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       2    IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       2    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       3    IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
       5    IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
       6    IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()

List of orphan blocks:
    Node    Method


Node: 1
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges to: 3 2
    Instructions:
        IL_0000: nop
        IL_0001: ldarg.1
        IL_0002: ldc.i4.2
        IL_0003: rem
        IL_0004: ldc.i4.0
        IL_0005: ceq
        IL_0007: stloc.0
        IL_0008: ldloc.0
        IL_0009: brfalse.s IL_001e


Node: 2
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 1
    Edges to: 4
    Instructions:
        IL_000b: ldarg.0
        IL_000c: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0011: ldarg.1
        IL_0012: ldarg.1
        IL_0013: ldc.i4.s 20
        IL_0015: mul
        IL_0016: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_001b: nop
        IL_001c: br.s IL_002f


Node: 3
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 1
    Edges to: 4
    Instructions:
        IL_001e: ldarg.0
        IL_001f: ldfld System.Collections.Generic.List`1<System.Int32> ConsoleApp4.Program/<>c__DisplayClass2_0::t1  
        IL_0024: ldarg.1
        IL_0025: ldarg.1
        IL_0026: ldc.i4.s 30
        IL_0028: mul
        IL_0029: callvirt System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
        IL_002e: nop


Node: 4
    Method System.Void ConsoleApp4.Program/<>c__DisplayClass2_0::<Main>b__0(System.Int32)
    HasThis   True
    Args   2
    Locals 1
    Return (reuse) False
    Edges from: 2 3
    Instructions:
        IL_002f: ret


Node: 5
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges to: 7 6
    Instructions:
        IL_0000: ldarg.1
        IL_0001: ldarg.0
        IL_0002: ldfld System.Int32 System.Collections.Generic.List`1<T>::_size
        IL_0007: blt.un.s IL_000e


Node: 6
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges from: 5
    Edges to: 7
    Instructions:
        IL_0000: call System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()


Node: 7
    Method System.Void System.Collections.Generic.List`1<System.Int32>::set_Item(System.Int32,!0)
    HasThis   True
    Args   3
    Locals 0
    Return (reuse) False
    Edges from: 5 6
    Instructions:
        IL_000e: ldarg.0
        IL_000f: ldfld T[] System.Collections.Generic.List`1<T>::_items
        IL_0014: ldarg.1
        IL_0015: ldarg.2
        IL_0016: stelem.any T
        IL_001b: ldarg.0
        IL_001c: ldarg.0
        IL_001d: ldfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0022: ldc.i4.1
        IL_0023: add
        IL_0024: stfld System.Int32 System.Collections.Generic.List`1<T>::_version
        IL_0029: ret


Node: 8
    Method System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()
    HasThis   False
    Args   0
    Locals 0
    Return (reuse) False
    Instructions:
        IL_0000: nop
        IL_0001: ret

Compile part 1, node 1
