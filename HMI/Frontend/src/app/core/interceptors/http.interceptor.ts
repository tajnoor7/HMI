import { HttpInterceptorFn } from '@angular/common/http';

export const httpInterceptor: HttpInterceptorFn = (req, next) => {
  
  const authReq = req.clone({
    setHeaders: {
      'Content-Type':  'application/json',
    }
  });
  return next(authReq);
};
