use std::ops::{Deref, DerefMut, Index, IndexMut};
mod iter;

/// A two-dimensional vector.
pub struct Vec2d<'a, T> {
    v: Vec2dSliceMut<'a, T>,
}

impl<T: Default + Clone> Vec2d<'_, T> {
    /// Create a new 2D vector with the given width and height.
    pub fn new(width: usize, height: usize) -> Self {
        let backing = vec![T::default(); width * height];
        Self {
            v: Vec2dSliceMut {
                v: backing.leak(),
                x_offset: 0,
                y_offset: 0,
                width,
                height,
                total_width: width,
                total_height: height,
            },
        }
    }
}

impl<'a, T> Deref for Vec2d<'a, T> {
    type Target = Vec2dSliceMut<'a, T>;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<'a, T> DerefMut for Vec2d<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

impl<T> Drop for Vec2d<'_, T> {
    fn drop(&mut self) {
        // SAFETY: we know that the pointer is valid because we created it
        unsafe {
            let _ = Vec::from_raw_parts(self.v.v.as_mut_ptr(), self.v.v.len(), self.v.v.len());
        };
    }
}

impl<T: Clone> Clone for Vec2d<'_, T> {
    fn clone(&self) -> Self {
        self.v.to_owned()
    }
}

/// A copyable immutable reference to (part of) a two-dimensional vector.
#[derive(Clone, Copy)]
pub struct Vec2dSlice<'a, T> {
    /// The backing 2D vector.
    v: &'a [T],
    /// The x offset of the slice in the backing 2D vector
    x_offset: usize,
    /// The y offset of the slice in the backing 2D vector
    y_offset: usize,
    /// The width of the slice
    width: usize,
    /// The height of the slice
    height: usize,
    /// The width of the entire backing 2D vector
    total_width: usize,
    /// The height of the entire backing 2D vector
    total_height: usize,
}

/// A mutable reference to (part of) a two-dimensional vector.
pub struct Vec2dSliceMut<'a, T> {
    /// The backing 2D vector.
    v: &'a mut [T],
    /// The x offset of the slice in the backing 2D vector
    pub x_offset: usize,
    /// The y offset of the slice in the backing 2D vector
    y_offset: usize,
    /// The width of the slice
    width: usize,
    /// The height of the slice
    height: usize,
    /// The width of the entire backing 2D vector
    total_width: usize,
    /// The height of the entire backing 2D vector
    total_height: usize,
}

impl<T: std::fmt::Debug> std::fmt::Debug for Vec2dSlice<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut chunks = Vec::with_capacity(self.height);
        for i in 0..self.height {
            let start_pos = (self.y_offset + i) * self.total_width + self.x_offset;
            chunks.push(&self.v[start_pos..start_pos + self.width]);
        }
        f.debug_struct("Vec2dSlice").field("v", &chunks).finish()
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Vec2dSliceMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut chunks = Vec::with_capacity(self.height);
        for i in 0..self.height {
            let start_pos = (self.y_offset + i) * self.total_width + self.x_offset;
            chunks.push(&self.v[start_pos..start_pos + self.width]);
        }
        f.debug_struct("Vec2dSliceMut").field("v", &chunks).finish()
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Vec2d<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vec2d")
            .field("v", &self.v.v.chunks_exact(self.width).collect::<Vec<_>>())
            .finish()
    }
}

impl<'a, T> Vec2dSlice<'a, T> {
    fn assert_x_in_bounds(&self, x: usize) {
        assert!(
            x < self.width,
            "out of bounds: x is {x} but width is {}",
            self.width
        );
    }

    fn assert_y_in_bounds(&self, y: usize) {
        assert!(
            y < self.height,
            "out of bounds: y is {y} but height is {}",
            self.height
        );
    }

    fn in_bounds(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// The total number of elements in the slice.
    pub fn size(&self) -> usize {
        self.width * self.height
    }

    /// Get a reference to the element at the given x and y coordinates.
    /// If the coordinates are out of bounds, `None` is returned.
    pub fn get(&self, x: usize, y: usize) -> Option<&'a T> {
        if !self.in_bounds(x, y) {
            return None;
        }
        // SAFETY: in_bounds ensures that the index is in bounds
        unsafe { Some(self.get_unchecked(x, y)) }
    }

    /// Get the element at the given x and y coordinates without bounds checks.
    ///
    /// # Safety
    ///
    /// The coordinates must be in bounds. Out of bounds access is undefined behavior.
    pub unsafe fn get_unchecked(&self, x: usize, y: usize) -> &'a T {
        self.v
            .get_unchecked((self.y_offset + y) * self.total_width + (self.x_offset + x))
    }

    /// Splits the slice into two slices at the given x offset.
    /// The left part covers the the x coordinates from 0 to `x_offset` (exclusive) and the right
    /// part from `x_offset` (inclusive) to `width` (exclusive).
    pub fn split_x(&mut self, x_offset: usize) -> (Vec2dSlice<'a, T>, Vec2dSlice<'a, T>) {
        self.assert_x_in_bounds(x_offset);

        let self_ptr = self.v as *const [T];
        // SAFETY: we ensured that the offset is in bounds and if the pointer was valid before it is still valid
        (
            Vec2dSlice {
                v: unsafe { &*self_ptr },
                x_offset: self.x_offset,
                y_offset: self.y_offset,
                width: x_offset,
                height: self.height,
                total_width: self.total_width,
                total_height: self.total_height,
            },
            Vec2dSlice {
                v: unsafe { &*self_ptr },
                x_offset: self.x_offset + x_offset,
                y_offset: self.y_offset,
                width: self.width - x_offset,
                height: self.height,
                total_width: self.total_width,
                total_height: self.total_height,
            },
        )
    }

    /// Splits the slice into two slices at the given y offset.
    /// The top part covers the the y coordinates from 0 to `y_offset` (exclusive) and the bottom
    /// part from `y_offset` (inclusive) to `height` (exclusive).
    pub fn split_y(&mut self, y_offset: usize) -> (Vec2dSlice<'a, T>, Vec2dSlice<'a, T>) {
        self.assert_y_in_bounds(y_offset);

        let self_ptr = self.v as *const [T];
        // SAFETY: we ensured that the offset is in bounds and if the pointer was valid before it is still valid
        (
            Vec2dSlice {
                v: unsafe { &*self_ptr },
                x_offset: self.x_offset,
                y_offset: self.y_offset,
                width: self.width,
                height: y_offset,
                total_width: self.total_width,
                total_height: self.total_height,
            },
            Vec2dSlice {
                v: unsafe { &*self_ptr },
                x_offset: self.x_offset,
                y_offset: self.y_offset + y_offset,
                width: self.width,
                height: self.height - y_offset,
                total_width: self.total_width,
                total_height: self.total_height,
            },
        )
    }

    /// An iterator over the references to the elements of the slice, row by row.
    pub fn iter<'slice>(&'slice self) -> iter::Iter<'a, 'slice, T> {
        iter::Iter::new(self)
    }

    /// Clones the contents of the slice to create a new 2D vector.
    pub fn to_owned<'b>(&self) -> Vec2d<'b, T>
    where
        T: Clone,
    {
        let vec = self.iter().cloned().collect::<Vec<_>>();
        assert!(vec.len() == self.size(), "incorrect size after clone");
        Vec2d {
            v: Vec2dSliceMut {
                v: vec.leak(),
                x_offset: 0,
                y_offset: 0,
                width: self.width,
                height: self.height,
                total_width: self.width,
                total_height: self.height,
            },
        }
    }
}

impl<'a, T> Deref for Vec2dSliceMut<'a, T> {
    type Target = Vec2dSlice<'a, T>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: we are transmuting a &mut T to a &T
        unsafe { std::mem::transmute(self) }
    }
}

impl<'a, T> Vec2dSliceMut<'a, T> {
    /// Get a mutable reference to the element at the given x and y coordinates.
    /// If the coordinates are out of bounds, `None` is returned.
    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&'a mut T> {
        if !self.in_bounds(x, y) {
            return None;
        }
        // SAFETY: in_bounds ensures that the index is in bounds
        unsafe { Some(self.get_unchecked_mut(x, y)) }
    }

    /// Get the element at the given x and y coordinates without bounds checks.
    ///
    /// # Safety
    ///
    /// The coordinates must be in bounds. Out of bounds access is undefined behavior.
    pub unsafe fn get_unchecked_mut(&mut self, x: usize, y: usize) -> &'a mut T {
        let val: &mut T = self
            .v
            .get_unchecked_mut((self.y_offset + y) * self.total_width + (self.x_offset + x));

        // SAFETY: We know that the slice as a lifetime of 'a
        std::mem::transmute::<&mut T, &'a mut T>(val)
    }

    /// Splits the slice into two mutable slices at the given x offset.
    /// The left part covers the the x coordinates from 0 to `x_offset` (exclusive) and the right
    /// part from `x_offset` (inclusive) to `width` (exclusive).
    pub fn split_x_mut(&mut self, x_offset: usize) -> (Vec2dSliceMut<'_, T>, Vec2dSliceMut<'_, T>) {
        self.assert_x_in_bounds(x_offset);

        let self_ptr = self.v as *mut [T];
        // SAFETY: we ensured that the offset is in bounds and if the pointer was valid before it is still valid
        (
            Vec2dSliceMut {
                v: unsafe { &mut *self_ptr },
                x_offset: self.x_offset,
                y_offset: self.y_offset,
                width: x_offset,
                height: self.height,
                total_width: self.total_width,
                total_height: self.total_height,
            },
            Vec2dSliceMut {
                v: unsafe { &mut *self_ptr },
                x_offset: self.x_offset + x_offset,
                y_offset: self.y_offset,
                width: self.width - x_offset,
                height: self.height,
                total_width: self.total_width,
                total_height: self.total_height,
            },
        )
    }

    /// Splits the slice into two mutable slices at the given y offset.
    /// The top part covers the the y coordinates from 0 to `y_offset` (exclusive) and the bottom
    /// part from `y_offset` (inclusive) to `height` (exclusive).
    pub fn split_y_mut(&mut self, y_offset: usize) -> (Vec2dSliceMut<'a, T>, Vec2dSliceMut<'a, T>) {
        self.assert_y_in_bounds(y_offset);

        let self_ptr = self.v as *mut [T];
        // SAFETY: we ensured that the offset is in bounds and if the pointer was valid before it is still valid
        (
            Vec2dSliceMut {
                v: unsafe { &mut *self_ptr },
                x_offset: self.x_offset,
                y_offset: self.y_offset,
                width: self.width,
                height: y_offset,
                total_width: self.total_width,
                total_height: self.total_height,
            },
            Vec2dSliceMut {
                v: unsafe { &mut *self_ptr },
                x_offset: self.x_offset,
                y_offset: self.y_offset + y_offset,
                width: self.width,
                height: self.height - y_offset,
                total_width: self.total_width,
                total_height: self.total_height,
            },
        )
    }

    /// An iterator over the mutable references to the elements of the slice, row by row.
    pub fn iter_mut<'slice>(&'slice mut self) -> iter::IterMut<'a, 'slice, T> {
        iter::IterMut::new(self)
    }
}

impl<T> Index<(usize, usize)> for Vec2dSlice<'_, T> {
    type Output = T;
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        self.assert_x_in_bounds(x);
        self.assert_y_in_bounds(y);
        // SAFETY: we ensured that the indices are in bounds
        unsafe { self.get_unchecked(x, y) }
    }
}

impl<T> Index<(usize, usize)> for Vec2dSliceMut<'_, T> {
    type Output = T;
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        self.assert_x_in_bounds(x);
        self.assert_y_in_bounds(y);
        // SAFETY: we ensured that the indices are in bounds
        unsafe { self.get_unchecked(x, y) }
    }
}

impl<T> IndexMut<(usize, usize)> for Vec2dSliceMut<'_, T> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        self.assert_x_in_bounds(x);
        self.assert_y_in_bounds(y);
        // SAFETY: we ensured that the indices are in bounds
        unsafe { self.get_unchecked_mut(x, y) }
    }
}

// ------- Eq ---------

impl<T: PartialEq> PartialEq for Vec2dSlice<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        if !(self.width == other.width && self.height == other.height) {
            return false;
        }

        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: PartialEq> PartialEq<Vec2dSliceMut<'_, T>> for Vec2dSlice<'_, T> {
    fn eq(&self, other: &Vec2dSliceMut<'_, T>) -> bool {
        self == &**other
    }
}

impl<T: PartialEq> PartialEq<Vec2d<'_, T>> for Vec2dSlice<'_, T> {
    fn eq(&self, other: &Vec2d<'_, T>) -> bool {
        self == &**other
    }
}

impl<T: Eq> Eq for Vec2dSlice<'_, T> {}

impl<T: PartialEq> PartialEq<Vec2dSlice<'_, T>> for Vec2dSliceMut<'_, T> {
    fn eq(&self, other: &Vec2dSlice<'_, T>) -> bool {
        &**self == other
    }
}

impl<T: PartialEq> PartialEq for Vec2dSliceMut<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: PartialEq> PartialEq<Vec2d<'_, T>> for Vec2dSliceMut<'_, T> {
    fn eq(&self, other: &Vec2d<'_, T>) -> bool {
        **self == **other
    }
}

impl<T: Eq> Eq for Vec2dSliceMut<'_, T> {}

impl<T: PartialEq> PartialEq<Vec2dSlice<'_, T>> for Vec2d<'_, T> {
    fn eq(&self, other: &Vec2dSlice<'_, T>) -> bool {
        &**self == other
    }
}

impl<T: PartialEq> PartialEq<Vec2dSliceMut<'_, T>> for Vec2d<'_, T> {
    fn eq(&self, other: &Vec2dSliceMut<'_, T>) -> bool {
        **self == **other
    }
}

impl<T: PartialEq> PartialEq for Vec2d<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: Eq> Eq for Vec2d<'_, T> {}
