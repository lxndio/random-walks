use super::{Vec2dSlice, Vec2dSliceMut};

pub struct Iter<'vals, 'slice, T> {
    slice: &'slice Vec2dSlice<'vals, T>,
    x: usize,
    y: usize,
}

impl<'vals, 'slice, T> Iter<'vals, 'slice, T> {
    pub(crate) fn new(slice: &'slice Vec2dSlice<'vals, T>) -> Self {
        Self { slice, x: 0, y: 0 }
    }
}

impl<'a, T> Iterator for Iter<'_, 'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let val = self.slice.get(self.x, self.y)?;

        self.x += 1;
        if self.x == self.slice.width() {
            self.x = 0;
            self.y += 1;
        }

        Some(val)
    }
}

pub struct IterMut<'vals, 'slice, T> {
    slice: &'slice mut Vec2dSliceMut<'vals, T>,
    x: usize,
    y: usize,
}

impl<'vals, 'slice, T> IterMut<'vals, 'slice, T> {
    pub(crate) fn new(slice: &'slice mut Vec2dSliceMut<'vals, T>) -> Self {
        Self { slice, x: 0, y: 0 }
    }
}

impl<'a, T> Iterator for IterMut<'_, 'a, T> {
    type Item = &'a mut T;

    fn next(&'_ mut self) -> Option<Self::Item> {
        let val = self.slice.get_mut(self.x, self.y)?;

        self.x += 1;
        if self.x == self.slice.width() {
            self.x = 0;
            self.y += 1;
        }

        Some(val)
    }
}
