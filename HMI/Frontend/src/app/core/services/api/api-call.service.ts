import { Injectable } from '@angular/core';
import { ConfigService } from '../config/config.service';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { finalize } from 'rxjs/operators';
@Injectable({
  providedIn: 'root'
})
export class ApiCallService {

  baseUrl: string = "";

  constructor(private configService: ConfigService,
    private httpClient: HttpClient
  ) {
      this.baseUrl = this.configService.baseURL
   }

   generateResponse(data: any): any {
    return this.httpClient.post(`${this.baseUrl}/ask`, data).pipe(
      map(res => res)
    );
  }

  // recentJobs(): any {
  //   return this.httpClient.get(`${this.baseUrl}/recent/`).pipe(map((res => res)))
  // }

  // getJobs(data: any): any {
  //   const params = new HttpParams({ fromObject: data });
  //   return this.httpClient.get(`${this.baseUrl}/search/`, { params }).pipe(map((res) => res));
  // }

  // getJobDetails(id: string): any {
  //   return this.httpClient.get(`${this.baseUrl}/jobs/${id}`).pipe(map((res => res)))
  // }
}
