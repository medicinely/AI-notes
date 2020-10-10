{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-1.06904497 -1.35873244  0.98058068]\n",
      " [-0.26726124  0.33968311  0.39223227]\n",
      " [ 1.33630621  1.01904933 -1.37281295]]\n"
     ]
    }
   ],
   "source": [
    "def stand():\n",
    "    \"\"\"\n",
    "    标准化缩放\n",
    "    \"\"\"\n",
    "    std = StandardScaler()\n",
    "    data = std.fit_transform([\n",
    "        [1.,-1.,3.],\n",
    "        [2.,4.,2.],\n",
    "        [4.,6.,-1.],\n",
    "    ])\n",
    "    print(data)\n",
    "    \n",
    "    return None\n",
    "stand()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
