package com.sih.document_screening.repository; 
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import com.sih.document_screening.model.Screening;

import java.util.Optional;
import java.util.UUID;

@Repository
public interface ScreeningRepository extends JpaRepository<Screening, UUID> {

    @Query("SELECT s FROM Screening s WHERE s.passportRecord.documentNumber = :passportNumber")
    Optional<Screening> findByPassportNumber(@Param("passportNumber") String passportNumber);

    @Query("SELECT s FROM Screening s WHERE s.nationalIdRecord.nationalIdNumber = :nationalIdNumber")
    Optional<Screening> findByNationalIdNumber(@Param("nationalIdNumber") String nationalIdNumber);
}