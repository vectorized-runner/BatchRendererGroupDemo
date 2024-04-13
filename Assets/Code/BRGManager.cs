using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using Random = Unity.Mathematics.Random;

namespace BRGDemo
{
	public unsafe class BRGManager : MonoBehaviour
	{
		public Mesh Mesh;
		public Material Material;
		public int SpawnCount = 100_000;
		public float SpawnRadius = 1_000.0f;
		
		private BatchRendererGroup _brg;
		private BatchMeshID _meshID;
		private BatchMaterialID _materialID;
		private GraphicsBuffer _graphicsBuffer;

		private const int kSizeOfMatrix = sizeof(float) * 4 * 4;
		private const int kSizeOfPackedMatrix = sizeof(float) * 4 * 3;
		private const int kSizeOfFloat4 = sizeof(float) * 4;
		private const int kBytesPerInstance = (kSizeOfPackedMatrix * 2) + kSizeOfFloat4;
		private const int kExtraBytes = kSizeOfMatrix * 2;
		private const int intSizeInBytes = 4;
		private BatchID m_BatchID;

		private void Start()
		{
			_brg = new BatchRendererGroup(OnPerformCulling, IntPtr.Zero);
			_meshID = _brg.RegisterMesh(Mesh);
			_materialID = _brg.RegisterMaterial(Material);
			_graphicsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Raw,
				BufferCountForInstances(kBytesPerInstance, (int)SpawnCount, kExtraBytes), intSizeInBytes);
			
			PopulateInstanceDataBuffer();
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
			var random = new Random(1);
			
			// Place a zero matrix at the start of the instance data buffer, so loads from address 0 return zero.
			var zero = new float4x4[] { new float4x4() };

			var matrices = new NativeArray<float4x4>(SpawnCount, Allocator.Temp);
			var objectToWorld = new NativeArray<PackedMatrix>(SpawnCount, Allocator.Temp);
			var worldToObject = new NativeArray<PackedMatrix>(SpawnCount, Allocator.Temp);
			var colors = new NativeArray<float4>(SpawnCount, Allocator.Temp);
			
			for (int i = 0; i < SpawnCount; i++)
			{
				var matrix = float4x4.TRS(random.NextFloat3Direction() * random.NextFloat() * SpawnRadius, quaternion.identity, new float3(1, 1, 1));
				matrices[i] = matrix;
				objectToWorld[i] = new PackedMatrix(matrix);
				worldToObject[i] = new PackedMatrix(math.inverse(matrix));
				colors[i] = new float4(random.NextFloat3(), 1.0f);
			}

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
			uint byteAddressWorldToObject = byteAddressObjectToWorld + kSizeOfPackedMatrix * (uint)SpawnCount;
			uint byteAddressColor = byteAddressWorldToObject + kSizeOfPackedMatrix * (uint)SpawnCount;

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
			// UnsafeUtility.Malloc() requires an alignment, so use the largest integer type's alignment
			// which is a reasonable default.
			int alignment = UnsafeUtility.AlignOf<long>();

			// Acquire a pointer to the BatchCullingOutputDrawCommands struct so you can easily
			// modify it directly.
			var drawCommands = (BatchCullingOutputDrawCommands*)cullingOutput.drawCommands.GetUnsafePtr();

			// Allocate memory for the output arrays. In a more complicated implementation, you would calculate
			// the amount of memory to allocate dynamically based on what is visible.
			// This example assumes that all of the instances are visible and thus allocates
			// memory for each of them. The necessary allocations are as follows:
			// - a single draw command (which draws SpawnCount instances)
			// - a single draw range (which covers our single draw command)
			// - SpawnCount visible instance indices.
			// You must always allocate the arrays using Allocator.TempJob.
			drawCommands->drawCommands =
				(BatchDrawCommand*)UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawCommand>(), alignment,
					Allocator.TempJob);
			drawCommands->drawRanges = (BatchDrawRange*)UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawRange>(),
				alignment, Allocator.TempJob);
			drawCommands->visibleInstances =
				(int*)UnsafeUtility.Malloc(SpawnCount * intSizeInBytes, alignment, Allocator.TempJob);
			drawCommands->drawCommandPickingInstanceIDs = null;

			drawCommands->drawCommandCount = 1;
			drawCommands->drawRangeCount = 1;
			drawCommands->visibleInstanceCount = SpawnCount;

			// This example doens't use depth sorting, so it leaves instanceSortingPositions as null.
			drawCommands->instanceSortingPositions = null;
			drawCommands->instanceSortingPositionFloatCount = 0;

			// Configure the single draw command to draw SpawnCount instances
			// starting from offset 0 in the array, using the batch, material and mesh
			// IDs registered in the Start() method. It doesn't set any special flags.
			drawCommands->drawCommands[0].visibleOffset = 0;
			drawCommands->drawCommands[0].visibleCount = (uint)SpawnCount;
			drawCommands->drawCommands[0].batchID = m_BatchID;
			drawCommands->drawCommands[0].materialID = _materialID;
			drawCommands->drawCommands[0].meshID = _meshID;
			drawCommands->drawCommands[0].submeshIndex = 0;
			drawCommands->drawCommands[0].splitVisibilityMask = 0xff;
			drawCommands->drawCommands[0].flags = 0;
			drawCommands->drawCommands[0].sortingPosition = 0;

			// Configure the single draw range to cover the single draw command which
			// is at offset 0.
			drawCommands->drawRanges[0].drawCommandsBegin = 0;
			drawCommands->drawRanges[0].drawCommandsCount = 1;

			// This example doesn't care about shadows or motion vectors, so it leaves everything
			// at the default zero values, except the renderingLayerMask which it sets to all ones
			// so Unity renders the instances regardless of mask settings.
			drawCommands->drawRanges[0].filterSettings = new BatchFilterSettings { renderingLayerMask = 0xffffffff, };

			// Finally, write the actual visible instance indices to the array. In a more complicated
			// implementation, this output would depend on what is visible, but this example
			// assumes that everything is visible.
			// for (int i = 0; i < SpawnCount; ++i)
			// {
			// 	drawCommands->visibleInstances[i] = i;
			// }

			// This simple example doesn't use jobs, so it returns an empty JobHandle.
			// Performance-sensitive applications are encouraged to use Burst jobs to implement
			// culling and draw command output. In this case, this function returns a
			// handle here that completes when the Burst jobs finish.
			
			return new CullJob
			{
				OutPtr = drawCommands,
				InstanceCount = SpawnCount,
			}.Schedule();
		}

		[BurstCompile]
		public struct CullJob : IJob
		{
			[NoAlias]
			[NativeDisableUnsafePtrRestriction]
			public BatchCullingOutputDrawCommands* OutPtr;

			public int InstanceCount;
			
			public void Execute()
			{
				var visibleIndex = 0;
				
				for (int i = 0; i < InstanceCount; i++)
				{
					// Render 1/1000
					if (i % 1000 == 0)
					{
						OutPtr->visibleInstances[visibleIndex++] = i;
					}
				}

				OutPtr->visibleInstanceCount = visibleIndex;
				OutPtr->drawCommands[0].visibleCount = (uint)visibleIndex;
			}
		}
	}
}