
**tf基础学习**

基本使用:  
1、使用图 (graph) 来表示计算任务。  

2、在被称之为 会话 (Session) 的上下文 (context) 中执行图。  

3、使用 tensor 表示数据。  

4、通过 变量 (Variable) 维护状态。  

5、使用 feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据。  

主要说一下feed 和 fetch ：  

**fetc**  
为了取回操作的输出内容, 可以在使用 Session 对象的 run() 调用 执行图时, 传入一些 tensor, 这些 tensor 会帮助你取回结果.   

在之前的例子里, 我们只取回了单个节点 state, 但是你也可以取回多个 tensor:  

**feed**  
TensorFlow 还提供了 feed 机制, 该机制 可以临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁, 直接插入一个 tensor.  

feed 使用一个 tensor 值临时替换一个操作的输出结果. 你可以提供 feed 数据作为 run()调用的参数. feed 只在调用它的方法内有效, 方法结束, feed 就会消失.  

最常见的用例是将某些特殊的操作指定为 "feed" 操作, 标记的方法是使用 tf.placeholder() 为这些操作创建占位符.
