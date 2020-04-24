# A PyTorch implementation of 2D SOM (Self Organizing Map) that runs fast.

The code is largely based on the original
[TensorFlow](https://www.tensorflow.org/)-implementation by Google engineer
Sachin Joglekar's in his
[blogpost](https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/).

## Screenshot
Here is the mandatory screenshot.

![Screenshot 1](https://raw.githubusercontent.com/theblackfly/fast-som/master/screenshots/1.png)

## Why?

Although I could find--at the time of this writing--another implementation of a
2D-SOM in [PyTorch](https://pytorch.org/) in [this
repository](https://github.com/giannisnik/som) which was also heavily based on
the original TensorFlow implementation, the PyTorch code ran too slowly.

So, here's an implementation that runs fast (relatively). Enjoy! :thumbsup:

## Usage

The `som.py` file contains the class definition of a 2D-SOM. Feel free to
tweak it to your needs.

The `colorsome.py` file showcases a nice example of using the `SOM2D` class from
`som.py` to cluster the RGB color space.
