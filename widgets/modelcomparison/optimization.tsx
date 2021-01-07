var tolerance = 1e-10;

function bisect(f, a, b) {
  if (f(a) == 0) {
      return a;
  } else if (f(b) == 0) {
      return b;
  } else if (f(a) * f(b) > 0) {
      throw new Error("f(a) and f(b) must evaluate to opposite signs.");
  }

  var mid = (a + b) / 2;
  for (var i=0;  i < 100; i++) {
    mid = (a + b) / 2;
    if (Math.abs(a - b) < tolerance) {
        break;
    }

    if (Math.abs(f(a)) < tolerance) {
        return a;
    } else if (Math.abs(f(b)) < tolerance) {
        return b;
    }

    if (f(mid) <= 0) {
      a = mid;
    } else if (f(mid) > 0) {
      b = mid;
    }
  }

  return mid;
}

export {bisect as bisect};
