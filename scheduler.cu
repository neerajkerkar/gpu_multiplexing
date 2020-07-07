#include <stdio.h> 
#include <sys/ipc.h> 
#include <sys/msg.h> 
#include <string.h>
#include <unistd.h>

#define TO_SCHEDULER 1
#define KERNEL_REGISTER 1
#define SLICE_CALLED 2

#define min(a, b) ((a) < (b)) ? (a) : (b)

typedef struct msg_buffer {
  long type;
  char content[50];
} msg_buf;

typedef struct kernel_struct {
  int tid;
  dim3 numBlocks;
  dim3 blockSize;
  dim3 nextBlockOffset;
  dim3 sliceDim;
} kernel;

typedef struct kernel_list_struct {
  kernel* kernel_array[5000];
  int size;
} kernel_list;

kernel_list kernels;

typedef struct scheduling_state_struct {
  int next_kernel_index;
} scheduling_state;

scheduling_state sched_state;

/*this needs improvement. currently slice size is chosen such that
  4 to 5 slices are created. */
void set_slice_size(dim3 *sliceSize, dim3 numBlocks, dim3 blockSize){
  int slices = 4;
  *sliceSize = numBlocks;
  int threads = numBlocks.x * numBlocks.y * numBlocks.z * blockSize.x * blockSize.y * blockSize.z;
  if(threads <= (1<<23)) return ;
  if(numBlocks.x >= slices){
    sliceSize->x = numBlocks.x/slices;
  } else if(numBlocks.y >= slices){
    sliceSize->y = numBlocks.y/slices;
  } else if(numBlocks.z >= slices){
    sliceSize->z = numBlocks.z/slices;
  }
}

kernel * new_kernel(int tid, dim3 numBlocks, dim3 blockSize){
  kernel *ker = (kernel *) malloc(sizeof(kernel));
  ker->tid = tid;
  ker->numBlocks = numBlocks;
  ker->blockSize = blockSize;
  ker->nextBlockOffset = {0,0,0};
  set_slice_size(&ker->sliceDim, numBlocks, blockSize);
  return ker;
}

void add_kernel(kernel_list *kernels, int tid, dim3 numBlocks, dim3 blockSize){
  kernels->kernel_array[kernels->size++] =  new_kernel(tid, numBlocks, blockSize);
}

void delete_kernel(kernel_list *kernels, int kern_indx){
  for(int i=kern_indx+1; i < kernels->size; i++){
    kernels->kernel_array[i-1] = kernels->kernel_array[i];
  }
  kernels->size = kernels->size - 1;
}

int get_num_kernels(kernel_list *kernels){
  return kernels->size;
}

int choose_next_slice(kernel_list *kernels, scheduling_state *sched_state, int *tid, dim3* sliceSize, dim3 *blockOffset, int *kernel_done){
  if(kernels->size == 0) return -1;
  kernel *next_kernel = kernels->kernel_array[sched_state->next_kernel_index];
  *tid = next_kernel->tid;
  *blockOffset = next_kernel->nextBlockOffset;
  dim3 numBlocks = next_kernel->numBlocks;
  dim3 sliceDim = next_kernel->sliceDim;
  sliceSize->x = min(sliceDim.x, numBlocks.x - blockOffset->x);
  sliceSize->y = min(sliceDim.y, numBlocks.y - blockOffset->y);
  sliceSize->z = min(sliceDim.z, numBlocks.z - blockOffset->z);
  dim3 nextBlockOffset = next_kernel->nextBlockOffset;
  nextBlockOffset.x += sliceDim.x;
  if(nextBlockOffset.x >= numBlocks.x){
    nextBlockOffset.x = 0;
    nextBlockOffset.y += sliceDim.y;
  }
  if(nextBlockOffset.y >= numBlocks.y){
    nextBlockOffset.y = 0;
    nextBlockOffset.z += sliceDim.z;
  }
  next_kernel->nextBlockOffset = nextBlockOffset;
  if(nextBlockOffset.z >= numBlocks.z){ //kernel done
    *kernel_done = 1;
    delete_kernel(kernels, sched_state->next_kernel_index);
    if(kernels->size > 0) sched_state->next_kernel_index = sched_state->next_kernel_index % kernels->size;
  }
  else {
    *kernel_done = 0;
    sched_state->next_kernel_index = (sched_state->next_kernel_index + 1) % kernels->size;
  }
  return 0;
}
    
  

void decode_kernel_registeration(msg_buf *msg, int *tid, dim3 *numBlocks, dim3 *blockSize){
  int nextByte = sizeof(int);
  memcpy(tid, msg->content + nextByte, sizeof(*tid));
  nextByte += sizeof(*tid);
  memcpy(numBlocks, msg->content + nextByte, sizeof(*numBlocks));
  nextByte += sizeof(*numBlocks);
  memcpy(blockSize, msg->content + nextByte, sizeof(*blockSize));
}

void decode_slice_called_msg(msg_buf *msg, int *tid){
  int nextByte = sizeof(int);
  memcpy(tid, msg->content + nextByte, sizeof(*tid));
}

void execute_slice(int msqid, int tid, dim3 sliceSize, dim3 blockOffset, int kernel_done){
  msg_buf msg;
  msg.type = tid;
  int nextByte=0;
  memcpy(msg.content + nextByte, &sliceSize, sizeof(sliceSize));
  nextByte += sizeof(sliceSize);
  memcpy(msg.content + nextByte, &blockOffset, sizeof(blockOffset));
  nextByte += sizeof(blockOffset);
  memcpy(msg.content + nextByte, &kernel_done, sizeof(kernel_done));
  nextByte += sizeof(kernel_done);
  msgsnd(msqid, &msg, nextByte, 0);
}

void schedule_next_slice(int msqid, kernel_list *kernels, scheduling_state *sched_state){
  int tid, kernel_done;
  dim3 sliceSize, blockOffset;
  int err = choose_next_slice(kernels, sched_state, &tid, &sliceSize, &blockOffset, &kernel_done);
  if(err != -1) execute_slice(msqid, tid, sliceSize, blockOffset, kernel_done);
}

int main(){
  key_t key;
  int msqid, subtype, tid;
  msg_buf msg;
  key = ftok("scheduler.c",19);
  msqid = msgget(key, 0666 | IPC_CREAT);
  while(msgrcv(msqid, &msg, sizeof(msg.content), TO_SCHEDULER, 0) >= 0){
    memcpy(&subtype, msg.content, sizeof(subtype));
    printf("subtype=%d\n", subtype);
    if(subtype ==  KERNEL_REGISTER){
      dim3 numBlocks, blockSize;
      decode_kernel_registeration(&msg, &tid, &numBlocks, &blockSize);
      printf("tid=%d, numblocks=%d %d %d, blockSize=%d %d %d\n",tid, numBlocks.x, numBlocks.y, numBlocks.z, blockSize.x, blockSize.y, blockSize.z);
      add_kernel(&kernels, tid, numBlocks, blockSize);
      if(get_num_kernels(&kernels)==1){
        schedule_next_slice(msqid, &kernels, &sched_state);
      }
    }
    else if(subtype == SLICE_CALLED){
      decode_slice_called_msg(&msg, &tid);
      printf("%d called slice\n",tid); 
      schedule_next_slice(msqid, &kernels, &sched_state);
    }
    else {
      printf("ERROR: Incorrect message subtype\n");
    }
  }
}
      
  

