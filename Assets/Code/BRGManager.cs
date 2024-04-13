using System;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Rendering;

namespace BRGDemo
{
	// The PackedMatrix is a convenience type that converts matrices into
	// the format that Unity-provided SRP shaders expect.
	public struct PackedMatrix
	{
		public float c0x;
		public float c0y;
		public float c0z;
		public float c1x;
		public float c1y;
		public float c1z;
		public float c2x;
		public float c2y;
		public float c2z;
		public float c3x;
		public float c3y;
		public float c3z;

		public PackedMatrix(Matrix4x4 m)
		{
			c0x = m.m00;
			c0y = m.m10;
			c0z = m.m20;
			c1x = m.m01;
			c1y = m.m11;
			c1z = m.m21;
			c2x = m.m02;
			c2y = m.m12;
			c2z = m.m22;
			c3x = m.m03;
			c3y = m.m13;
			c3z = m.m23;
		}
	}

	public class BRGManager : MonoBehaviour
	{
		public Mesh Mesh;
		public Material Material;

		private BatchRendererGroup _brg;
		private BatchMeshID _meshID;
		private BatchMaterialID _materialID;
		private GraphicsBuffer _graphicsBuffer;

		private const int kSizeOfMatrix = sizeof(float) * 4 * 4;
		private const int kSizeOfPackedMatrix = sizeof(float) * 4 * 3;
		private const int kSizeOfFloat4 = sizeof(float) * 4;
		private const int kBytesPerInstance = (kSizeOfPackedMatrix * 2) + kSizeOfFloat4;
		private const int kExtraBytes = kSizeOfMatrix * 2;
		private const int kNumInstances = 3;
		private const int intSizeInBytes = 4;
		private BatchID m_BatchID;

		private void Start()
		{
			_brg = new BatchRendererGroup(OnPerformCulling, IntPtr.Zero);
			_meshID = _brg.RegisterMesh(Mesh);
			_materialID = _brg.RegisterMaterial(Material);
			_graphicsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Raw,
				BufferCountForInstances(kBytesPerInstance, kNumInstances, kExtraBytes), intSizeInBytes);
		}

		// Raw buffers are allocated in ints. This is a utility method that calculates
		// the required number of ints for the data.
		static int BufferCountForInstances(int bytesPerInstance, int numInstances, int extraBytes = 0)
		{
			// Round byte counts to int multiples
			bytesPerInstance = (bytesPerInstance + intSizeInBytes - 1) / intSizeInBytes * intSizeInBytes;
			extraBytes = (extraBytes + intSizeInBytes - 1) / intSizeInBytes * intSizeInBytes;
			int totalBytes = bytesPerInstance * numInstances + extraBytes;
			return totalBytes / intSizeInBytes;
		}

		private void OnDestroy()
		{
			_brg.Dispose();
		}

		private void PopulateInstanceDataBuffer()
		{
			// Place a zero matrix at the start of the instance data buffer, so loads from address 0 return zero.
			var zero = new Matrix4x4[1] { Matrix4x4.zero };

			// Create transform matrices for three example instances.
			var matrices = new Matrix4x4[kNumInstances]
			{
				Matrix4x4.Translate(new Vector3(-2, 0, 0)),
				Matrix4x4.Translate(new Vector3(0, 0, 0)),
				Matrix4x4.Translate(new Vector3(2, 0, 0)),
			};

			// Convert the transform matrices into the packed format that the shader expects.
			var objectToWorld = new PackedMatrix[kNumInstances]
			{
				new PackedMatrix(matrices[0]),
				new PackedMatrix(matrices[1]),
				new PackedMatrix(matrices[2]),
			};

			// Also create packed inverse matrices.
			var worldToObject = new PackedMatrix[kNumInstances]
			{
				new PackedMatrix(matrices[0].inverse),
				new PackedMatrix(matrices[1].inverse),
				new PackedMatrix(matrices[2].inverse),
			};

			// Make all instances have unique colors.
			var colors = new Vector4[kNumInstances]
			{
				new Vector4(1, 0, 0, 1),
				new Vector4(0, 1, 0, 1),
				new Vector4(0, 0, 1, 1),
			};

			// In this simple example, the instance data is placed into the buffer like this:
			// Offset | Description
			//      0 | 64 bytes of zeroes, so loads from address 0 return zeroes
			//     64 | 32 uninitialized bytes to make working with SetData easier, otherwise unnecessary
			//     96 | unity_ObjectToWorld, three packed float3x4 matrices
			//    240 | unity_WorldToObject, three packed float3x4 matrices
			//    384 | _BaseColor, three float4s

			// Calculates start addresses for the different instanced properties. unity_ObjectToWorld starts
			// at address 96 instead of 64, because the computeBufferStartIndex parameter of SetData
			// is expressed as source array elements, so it is easier to work in multiples of sizeof(PackedMatrix).
			uint byteAddressObjectToWorld = kSizeOfPackedMatrix * 2;
			uint byteAddressWorldToObject = byteAddressObjectToWorld + kSizeOfPackedMatrix * kNumInstances;
			uint byteAddressColor = byteAddressWorldToObject + kSizeOfPackedMatrix * kNumInstances;

			// Upload the instance data to the GraphicsBuffer so the shader can load them.
			_graphicsBuffer.SetData(zero, 0, 0, 1);
			_graphicsBuffer.SetData(objectToWorld, 0, (int)(byteAddressObjectToWorld / kSizeOfPackedMatrix),
				objectToWorld.Length);
			_graphicsBuffer.SetData(worldToObject, 0, (int)(byteAddressWorldToObject / kSizeOfPackedMatrix),
				worldToObject.Length);
			_graphicsBuffer.SetData(colors, 0, (int)(byteAddressColor / kSizeOfFloat4), colors.Length);

			// Set up metadata values to point to the instance data. Set the most significant bit 0x80000000 in each
			// which instructs the shader that the data is an array with one value per instance, indexed by the instance index.
			// Any metadata values that the shader uses that are not set here will be 0. When a value of 0 is used with
			// UNITY_ACCESS_DOTS_INSTANCED_PROP (i.e. without a default), the shader interprets the
			// 0x00000000 metadata value and loads from the start of the buffer. The start of the buffer is
			// a zero matrix so this sort of load is guaranteed to return zero, which is a reasonable default value.
			var metadata = new NativeArray<MetadataValue>(3, Allocator.Temp);
			metadata[0] = new MetadataValue
				{ NameID = Shader.PropertyToID("unity_ObjectToWorld"), Value = 0x80000000 | byteAddressObjectToWorld, };
			metadata[1] = new MetadataValue
				{ NameID = Shader.PropertyToID("unity_WorldToObject"), Value = 0x80000000 | byteAddressWorldToObject, };
			metadata[2] = new MetadataValue
				{ NameID = Shader.PropertyToID("_BaseColor"), Value = 0x80000000 | byteAddressColor, };

			// Finally, create a batch for the instances and make the batch use the GraphicsBuffer with the
			// instance data as well as the metadata values that specify where the properties are.
			m_BatchID = _brg.AddBatch(metadata, _graphicsBuffer.bufferHandle);
		}


		public unsafe JobHandle OnPerformCulling(
			BatchRendererGroup rendererGroup,
			BatchCullingContext cullingContext,
			BatchCullingOutput cullingOutput,
			IntPtr userContext)
		{
			// This example doesn't use jobs, so it can return an empty JobHandle.
			// Performance-sensitive applications should use Burst jobs to implement
			// culling and draw command output. In this case, this function would return a
			// handle here that completes when the Burst jobs finish.
			return new JobHandle();
		}
	}
}