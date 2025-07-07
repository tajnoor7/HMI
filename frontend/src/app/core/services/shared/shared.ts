import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class Shared {

  private loadingSource = new BehaviorSubject<boolean>(false);
  private resultSource = new BehaviorSubject<any>(null);

  // Observables for components to subscribe
  loading$ = this.loadingSource.asObservable();
  result$ = this.resultSource.asObservable();

  // Methods to update the states


  setResult(data: any): void {
    this.resultSource.next(data);
  }
}
