#include <sys/syscall.h>
#include <sys/ipc.h> 
#include <sys/msg.h>
#include <unistd.h>

struct scheduling_info {
  int msqid;
} sched_info = {-1};

typedef struct msg_buffer {
  long type;
  char content[50];
} msg_buf;

#define TO_SCHEDULER 1
#define KERNEL_REGISTER 1
#define SLICE_CALLED 2

inline void register_kernel(int tid,dim3 numBlocks,dim3 blockSize){\
  msg_buf msg;
  msg.type = TO_SCHEDULER;
  int subtype = KERNEL_REGISTER;
  char* nextFreeSpace = msg.content;
  memcpy(nextFreeSpace, &subtype, sizeof(subtype));
  nextFreeSpace += sizeof(subtype);
  memcpy(nextFreeSpace, &tid, sizeof(tid));
  nextFreeSpace += sizeof(tid);
  memcpy(nextFreeSpace, &numBlocks, sizeof(numBlocks));
  nextFreeSpace += sizeof(numBlocks);
  memcpy(nextFreeSpace, &blockSize, sizeof(blockSize));
  nextFreeSpace += sizeof(blockSize);
  msgsnd(sched_info.msqid, &msg, nextFreeSpace - msg.content, 0);
}

inline void get_next_slice(int tid, dim3 *sliceSize, dim3 *blockOffset, int *kernel_done){
  msg_buf msg;
  msgrcv(sched_info.msqid, &msg, sizeof(msg.content), tid, 0);
  memcpy(sliceSize, msg.content, sizeof(*sliceSize));
  memcpy(blockOffset, msg.content + sizeof(*sliceSize), sizeof(*blockOffset));
  memcpy(kernel_done, msg.content + sizeof(*sliceSize) + sizeof(*blockOffset), sizeof(*kernel_done));
}
  
inline void slice_called(int tid){
  //std::cout << "In slice called with tid: " << tid << std::endl;
  msg_buf msg;
  msg.type = TO_SCHEDULER;
  int subtype = SLICE_CALLED;
  char *nextFreeSpace = msg.content;
  memcpy(nextFreeSpace, &subtype, sizeof(subtype));
  nextFreeSpace += sizeof(subtype);
  memcpy(nextFreeSpace, &tid, sizeof(tid));
  nextFreeSpace += sizeof(tid);
  msgsnd(sched_info.msqid, &msg, nextFreeSpace - msg.content, 0);
}
  

#define SLICER(numBlocks, blockSize, kernel_name, kernel_args...) {\
  int tid = (pid_t) syscall (SYS_gettid);\
  if(sched_info.msqid==-1){\
    key_t key = ftok("scheduler.c",19);\
    sched_info.msqid = msgget(key, 0666 | IPC_CREAT);\
  }\
  register_kernel(tid, numBlocks, blockSize);\
  dim3 sliceSize;\
  dim3 blockOffset;\
  int kernel_done;\
  do {\
    get_next_slice(tid, &sliceSize, &blockOffset, &kernel_done);\
    kernel_name<<<sliceSize, blockSize>>>(blockOffset, numBlocks, kernel_args);\
    cudaStreamSynchronize(0);\
    slice_called(tid);\
  } while(kernel_done == 0);\
}

