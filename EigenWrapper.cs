using System.Runtime.InteropServices;


public class EigenWrapper
{
    [DllImport("EigenWrapper", CallingConvention = CallingConvention.Cdecl)]
    public static extern float registerIsotropic([MarshalAs(UnmanagedType.LPArray)] float[] X, [MarshalAs(UnmanagedType.LPArray)] float[] Y, int N, 
        [MarshalAs(UnmanagedType.LPArray)] float[] outR, [MarshalAs(UnmanagedType.LPArray)] float[] outT);

    [DllImport("EigenWrapper", CallingConvention = CallingConvention.Cdecl)]
    public static extern float registerAnisotropic([MarshalAs(UnmanagedType.LPArray)] float[] X, [MarshalAs(UnmanagedType.LPArray)] float[] Y, [MarshalAs(UnmanagedType.LPArray)] float[] W, int N, float threshold,
    [MarshalAs(UnmanagedType.LPArray)] float[] outR, [MarshalAs(UnmanagedType.LPArray)] float[] outT);

}