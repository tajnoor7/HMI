import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { SubSink } from 'subsink';
import { ApiCallService } from '../../services/api/api-call.service';
import { Shared } from '../../services/shared/shared';

@Component({
  selector: 'app-sidebar',
  imports: [],
  templateUrl: './sidebar.html',
  styleUrl: './sidebar.scss'
})
export class Sidebar {
  subs = new SubSink();
  response: any = [];

  constructor(public apiService: ApiCallService, 
    private http: HttpClient,
    private sharedService: Shared
  ) {
   }

   generateResponse(e: any): void {
     let text: String = e.target.innerText
    this.sharedService.setResult({text, isLoading: true});
  }
  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }

}
