import tensorflow as tf

def scharr_filter(input):
  '''
    return an scharr grdient image of the input
    input: a tensor with 4 dimensions [batch, height, width, channel]
    '''
  scharrx = tf.constant([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]])
  scharry = tf.constant([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]])
  scharr_x = tf.reshape(scharrx, [3,3,1,1])
  scharr_y = tf.reshape(scharry, [3,3,1,1])
  input_r = tf.reshape(input[:,:,:,0],[-1,input.shape[1],input.shape[2],1])
  input_g = tf.reshape(input[:,:,:,1],[-1,input.shape[1],input.shape[2],1])
  input_b = tf.reshape(input[:,:,:,2],[-1,input.shape[1],input.shape[2],1])
  gradx_r = conv2d(input_r, scharr_x)
  grady_r = conv2d(input_r, scharr_y)
  gradxy_r = 0.5*gradx_r + 0.5*grady_r
  gradx_g = conv2d(input_g, scharr_x)
  grady_g = conv2d(input_g, scharr_y)
  gradxy_g = 0.5*gradx_g + 0.5*grady_g
  gradx_b = conv2d(input_b, scharr_x)
  grady_b = conv2d(input_b, scharr_y)
  gradxy_b = 0.5*gradx_b + 0.5*grady_b
  gradxy = tf.concat([gradxy_r, gradxy_g], axis=3)
  gradxy = tf.concat([gradxy, gradxy_b], axis=3)
  return gradxy
