import { CommonModule } from '@angular/common';
import { Component, HostListener } from '@angular/core';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.scss'
})
export class NavbarComponent {

  isLanguageClicked: boolean = false;
  activeLanguage: string = "English";
  languageList: any = [
    {
      name: "English",
      isActive: true
    },
    {
      name: "German",
      isActive: false
    },
    {
      name: "Bengali",
      isActive: false
    }
  ]

  ngOnInit(): void {}

  @HostListener('window:scroll', ['$event'])
  onWindowScroll() {
    // let element = document.querySelector('.navbar') as HTMLElement;
    // if (window.pageYOffset > element.clientHeight) {
    //   element.classList.add('navbar-scroll');
    // } else {
    //   element.classList.remove('navbar-scroll');
    // }
  }

  clickLanguageSwitcher(): void {
    this.isLanguageClicked = !this.isLanguageClicked;
  }

}
