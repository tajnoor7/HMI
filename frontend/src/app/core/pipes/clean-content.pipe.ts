import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'cleanContent'
})
export class CleanContentPipe implements PipeTransform {
  transform(value: string): string {
    if (!value) return '';
    
    // Remove "Intent:" (case-insensitive), trim remaining
    return value.replace(/^Intent:\s*/i, '').trim();
  }
}
